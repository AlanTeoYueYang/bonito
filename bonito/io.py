"""
Bonito Input/Output
"""

import os
import sys
from glob import glob
from textwrap import wrap
from multiprocessing import Process, Queue

from tqdm import tqdm
import numpy as np

import h5py

from bonito.decode import decode, decode_revised
from bonito.util import get_raw_data


class PreprocessReader(Process):
    """
    Reader Processor that reads and processes fast5 files
    """
    def __init__(self, directory, maxsize=5):
        super().__init__()
        self.directory = directory
        self.queue = Queue(maxsize)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        for fast5 in tqdm(glob("%s/*fast5" % self.directory), ascii=True, ncols=100):
            for read_id, raw_data in get_raw_data(fast5):
                self.queue.put((read_id, raw_data))
        self.queue.put(None)

    def stop(self):
        self.join()

class DecoderWriterRevised(Process):
    """
    Decoder Process that writes fasta records to stdoutx`
    """
    def __init__(self, alphabet, beamsize, kmer_length, hdf5_filename, wrap=100):
        super().__init__()
        self.queue = Queue()
        self.wrap = wrap
        self.beamsize = beamsize
        self.alphabet = ''.join(alphabet)
        self.kmer_length = kmer_length
        self.hdf5_filename = hdf5_filename

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        hdf5_file = h5py.File(self.hdf5_filename, 'w')
        hdf5_file.create_group('Reads')
        reads = hdf5_file['Reads']
        while True:
            job = self.queue.get()
            if job is None: return
            read_id, predictions, signal_data = job
            sequence, means = decode_revised(predictions, self.alphabet, signal_data, self.kmer_length, self.beamsize)
            if len(means) > 0:
#                 sys.stderr.write("\n> No. of kmers: {}\n".format(len(means)))
                reads.create_group(read_id)
                reads[read_id]['means'] = means
            sys.stdout.write(">%s\n" % read_id)
            sys.stdout.write("%s\n" % os.linesep.join(wrap(sequence, self.wrap)))
            sys.stdout.flush()
        hdf5_file.close()

    def stop(self):
        self.queue.put(None)
        self.join()


class DecoderWriter(Process):
    """
    Decoder Process that writes fasta records to stdout
    """
    def __init__(self, alphabet, beamsize=5, wrap=100):
        super().__init__()
        self.queue = Queue()
        self.wrap = wrap
        self.beamsize = beamsize
        self.alphabet = ''.join(alphabet)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        while True:
            job = self.queue.get()
            if job is None: return
            read_id, predictions = job
            sequence = decode(predictions, self.alphabet, self.beamsize)
            sys.stdout.write(">%s\n" % read_id)
            sys.stdout.write("%s\n" % os.linesep.join(wrap(sequence, self.wrap)))
            sys.stdout.flush()

    def stop(self):
        self.queue.put(None)
        self.join()
