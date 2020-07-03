import numpy as np
from scipy.special import rel_entr
from tqdm import tqdm
#from simplegoodturing import simpleGoodTuringProbs
from simple_good_turing import Estimator

from dssm.config import NO_OF_TRIGRAMS

from batchiterators.fileiterators import WikiQAFileIterator, ReutersFileIterator, SquadFileIterator, QuoraFileIterator, \
    FileIterator

wikiqa_iter = WikiQAFileIterator(
    "datasets/wikiqa/data.csv",
    "datasets/wikiqa/data.csv",
    batch_size=1,
    no_of_irrelevant_samples=0,
    encodingType="NGRAM",
    dense=True,
    shuffle=False
)

squad_iter = SquadFileIterator(
    "datasets/squad/data.csv",
    "datasets/squad/data.csv",
    batch_size=1,
    no_of_irrelevant_samples=0,
    encodingType="NGRAM",
    dense=True,
    shuffle=False
)

rcv1_iter = ReutersFileIterator(
    "datasets/rcv1/total.json",
    "datasets/rcv1/total.json",
    batch_size=1,
    no_of_irrelevant_samples=0,
    encodingType="NGRAM",
    dense=True
)

quora_iter = QuoraFileIterator(
    "datasets/quora/data.csv",
    "datasets/quora/trainval.csv",
    batch_size=1,
    no_of_irrelevant_samples=0,
    encodingType="NGRAM",
    dense=True,
    shuffle=False
)


def freq_of_freqs(freqs):
    N = {}
    for freq in freqs:
        freq = int(freq)
        N[freq] = N.get(freq, 0) + 1

    N[0] = 0
    for freq in range(max(N.keys())):
        if freq not in N:
            N[freq] = 0

    return N


def create_good_turing_probs(freqs):
    N = freq_of_freqs(freqs)
    #N = create_ngram_dict()  # Should be a dict like {ngram_idx_1: freq_1, ngram_idx_2: freq_2, ...} #sgt.sgt.ChinesePluralsTest.input

    good_turing_probs = Estimator(N=N).Z
    output = []
    for i, freq in enumerate(freqs):
        estimated_probability = good_turing_probs[i+1]
        output.append(estimated_probability)
    return np.array(output)


def get_probs(iterator: FileIterator):
    freqs = np.zeros(NO_OF_TRIGRAMS, np.float64)
    for batch in tqdm(iterator):
        q = batch.get_q_dense()[0]
        d = batch.get_relevant_dense()[0]
        freqs += q
        freqs += d
    #freqs = create_good_turing_probs(freqs)
    freqs += 1
    return normalize(freqs)


def normalize(vec: np.ndarray):
    sum = np.sum(vec)
    vec /= sum
    return vec


def kl_div(p, q):
    kl_div = rel_entr(p, q)
    return sum(kl_div)


wikiqa_probs = get_probs(wikiqa_iter)
squad_probs = get_probs(squad_iter)
rcv1_probs = get_probs(rcv1_iter)
quora_probs = get_probs(quora_iter)

print(wikiqa_probs)
print(squad_probs)
print(rcv1_probs)
print(quora_probs)
print()

print(kl_div(wikiqa_probs, squad_probs))
print(kl_div(squad_probs, wikiqa_probs))
print()

print(kl_div(wikiqa_probs, rcv1_probs))
print(kl_div(rcv1_probs, wikiqa_probs))
print()

print(kl_div(wikiqa_probs, quora_probs))
print(kl_div(quora_probs, wikiqa_probs))
print()

