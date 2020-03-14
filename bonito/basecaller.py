"""
Bonito Basecaller
"""

import sys
import time
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bonito.util import load_model, get_raw_data
from bonito.decode import decode_revised
import h5py
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from glob import glob
from textwrap import wrap

import torch
import numpy as np

def handle_output_directory(output_dir):
    """
    Process the output directory and return a valid directory where we save the output
    :param output_dir: Output directory path
    :return:
    """
    timestr = time.strftime("%m%d%Y_%H%M%S")
    # process the output directory
    if output_dir[-1] != "/":
        output_dir += "/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir

def setup(rank, total_gpu):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=total_gpu)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    # torch.manual_seed(42)

def cleanup():
    dist.destroy_process_group()

def basecall(rank, total_gpu, args, input_files):
    setup(rank, total_gpu)

    device_id = rank
    sys.stderr.write("INFO: LOADING MODEL ON DEVICE: {}\n".format(device_id))
    model = load_model(args.model_directory, args.device, weights=int(args.weights), half=args.half)
    torch.cuda.set_device(device_id)
    model.to(device_id)
    model.eval()
    model = DDP(model, device_ids=[device_id])
    sys.stderr.write("INFO: LOADED MODEL ON DEVICE: {}\n".format(device_id))

    samples = 0
    num_reads = 0
    max_read_size = 1e9
    dtype = np.float16 if args.half else np.float32
    sys.stderr.write('No of files:{}, index: {}'.format(len(input_files), rank))
    hdf5_file = open('{}_{}.hdf5'.format(args.prefix, device_id), 'w')
    fasta_file = open('{}_{}.fasta'.format(args.prefix, device_id), 'w')

    output_directory = handle_output_directory(os.path.abspath(args.output_directory))

    t0 = time.perf_counter()
    sys.stderr.write("STARTING INFERENCE\n")
    st = time.time()
    for fast5 in input_files:
        for read_id, raw_data in get_raw_data(fast5):
            num_reads += 1
            samples += len(raw_data)
            signal_data = raw_data

            raw_data = raw_data[np.newaxis, np.newaxis, :].astype(dtype)
            gpu_data = torch.tensor(raw_data).to(args.device)   
            posteriors = model(gpu_data).exp().cpu().numpy().squeeze()

            sequence, means = decode_revised(posteriors, model.alphabet, signal_data, args.kmer_length, args.beamsize)
            if len(means) > 0:
                sys.stderr.write("\n> No. of kmers: {}\n".format(len(means)))
                reads.create_group(read_id)
                reads[read_id]['means'] = means
            fasta_file.write(">%s\n" % read_id)
            fasta_file.write("%s\n" % os.linesep.join(wrap(sequence, 100)))

        ct = time.time()
        sys.stderr.write("\nINFO: FINISHED PROCESSING: {}/{} FILES. DEVICE: {} ELAPSED TIME: {}".format(count, len(input_files), device_id, ct-st))

    t1 = time.perf_counter()
    sys.stderr.write("INFO: TOTAL READS: %s\n" % num_reads)
    sys.stderr.write("INFO: TOTAL DURATION %.1E\n" % (t1 - t0))
    sys.stderr.write("INFO: SAMPLES PER SECOND %.1E\n" % (count/(t1 - t0)))
    sys.stderr.write("DONE\n")

    cleanup()

def main(args):
    total_gpu = torch.cuda.device_count()

    input_files = glob("%s/*fast5" % args.reads_directory, recursive=True)
    chunk_length = int(len(input_files) / total_gpu)
    file_chunks = []
    for i in range(0, len(input_files), chunk_length):
        file_chunks.append(input_files[i:i + chunk_length])

    mp.spawn(basecall,
             args=(total_gpu, args, file_chunks),
             nprocs=total_gpu,
             join=True)

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("model_directory")
    parser.add_argument("reads_directory")
    parser.add_argument("output_directory")
    parser.add_argument("prefix")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default="0", type=str)
    parser.add_argument("--alphabet", default="NACGT", type=str)
    parser.add_argument("--beamsize", default=5, type=int)
    parser.add_argument("--kmer_length", default=5, type=int)
    parser.add_argument("--half", action="store_true", default=False)
    return parser
