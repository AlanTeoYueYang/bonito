"""
Bonito Basecaller
"""

import sys
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bonito.util import load_model
# from bonito.io import DecoderWriter, PreprocessReader
from bonito.io import DecoderWriterRevised, PreprocessReader
from bonito.decode import decode_revised
import h5py

import torch
import numpy as np


def main(args):

	sys.stderr.write("> loading model\n")
	model = load_model(args.model_directory, args.device, weights=int(args.weights), half=args.half)

	samples = 0
	num_reads = 0
	max_read_size = 1e9
	dtype = np.float16 if args.half else np.float32
	reader = PreprocessReader(args.reads_directory)
	writer = DecoderWriterRevised(model.alphabet, args.beamsize, args.kmer_length, args.hdf5_filename)
	# writer = DecoderWriter(model.alphabet, args.beamsize)

	t0 = time.perf_counter()
	# sys.stderr.write("> calling\n")

	# with reader, torch.no_grad():
	with writer, reader, torch.no_grad():

		while True:

			read = reader.queue.get()
			if read is None:
				break

			read_id, raw_data = read
			if len(raw_data) > max_read_size:
				sys.stderr.write("> skipping %s: %s too long\n" % (len(raw_data), read_id))
				pass
			num_reads += 1
			samples += len(raw_data)
			signal_data = raw_data

			raw_data = raw_data[np.newaxis, np.newaxis, :].astype(dtype)
			gpu_data = torch.tensor(raw_data).to(args.device)	
			posteriors = model(gpu_data).exp().cpu().numpy().squeeze()

			# writer.queue.put((read_id, posteriors))

			# sys.stderr.write("\n> idx: %s\tcurrent read: %s" % (num_reads, read_id))
				
			writer.queue.put((read_id, posteriors, signal_data))
			
	duration = time.perf_counter() - t0

	sys.stderr.write("> completed reads: %s\n" % num_reads)
	sys.stderr.write("> total duration : %ss\n" % duration)
	sys.stderr.write("> samples per second %.1E\n" % (samples  / duration))
	sys.stderr.write("> done\n")


def argparser():
	parser = ArgumentParser(
		formatter_class=ArgumentDefaultsHelpFormatter,
		add_help=False
	)
	parser.add_argument("model_directory")
	parser.add_argument("reads_directory")
	parser.add_argument("hdf5_filename")
	parser.add_argument("--device", default="cuda")
	parser.add_argument("--weights", default="0", type=str)
	parser.add_argument("--beamsize", default=5, type=int)
	parser.add_argument("--kmer_length", default=5, type=int)
	parser.add_argument("--distributed", default=False, type=bool)
	parser.add_argument("--half", action="store_true", default=False)
	return parser
