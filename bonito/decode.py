"""
Bonito Decoding functions
"""

from itertools import groupby

import numpy as np
from fast_ctc_decode import beam_search
import sys

def decode_ref(encoded, labels):
	"""
	Convert a integer encoded reference into a string and remove blanks
	"""
	return ''.join(labels[e] for e in encoded if e)


def greedy_ctc_decode(predictions, labels):
	"""
	Greedy argmax decoder with collapsing repeats
	"""
	path = np.argmax(predictions, axis=1)
	return ''.join([labels[b] for b, g in groupby(path) if b])


def decode(predictions, alphabet, beam_size=5, threshold=0.1):
	"""
	Decode model posteriors to sequence
	"""
	alphabet = ''.join(alphabet)
	if beam_size == 1:
		return greedy_ctc_decode(predictions, alphabet)
	return beam_search(predictions.astype(np.float32), alphabet, beam_size, threshold)


def decode_revised(predictions, alphabet, signal_data, kmer_length=5, beam_size=5, threshold=0.1):
	"""
	Decode model posteriors to sequence
	"""
	alphabet = ''.join(alphabet)
	if beam_size == 1:
		return greedy_ctc_decode(predictions, alphabet)
	seq, path = beam_search(predictions.astype(np.float32), alphabet, beam_size, threshold)
	means = []
	if len(path) > 0:
		if path[0] != 0: path = [0] + path
		if path[:-1] != len(signal_data): path.append(len(signal_data))
		if kmer_length < len(seq):
			for i in range(len(seq)-kmer_length+1):
				start_idx, end_idx = path[i], path[i+kmer_length]
				mean = np.mean(signal_data[start_idx:end_idx])
				means.append(mean)
			min_v, max_v = np.min(means), np.max(means)
			for j in range(len(means)):
				means[j] -= min_v
				means[j] /= (max_v-min_v)
				means[j] *= 255
				means[j] = means[j].astype('uint8')
		else:
			means.append(0)
	return seq, np.asarray(means)
