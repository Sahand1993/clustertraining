# Calculate correlation within pretraining data set (treat as one)

from batchiterators.batchiterators import RandomBatchIterator
from batchiterators.fileiterators import *
import random
from scipy.stats import pearsonr

def pearsoncoeff(iterator, n):
    avg_pearsonr = 0
    for i in range(n):
        if i % 10000 == 0:
            print("{} pairs looked at".format(i))
        a = iterator.__next__()
        b = iterator.__next__()
        a_vec = a.get_q_dense() if random.randint(0, 1) else a.get_relevant_dense()
        b_vec = b.get_q_dense() if random.randint(0, 1) else b.get_relevant_dense()
        r, _ = pearsonr(a_vec[0], b_vec[0])
        avg_pearsonr += r / (n)
    return avg_pearsonr

quoraTrainingSet = QuoraFileIterator(
          "datasets_smallnq/quora/total.csv",
          "datasets_smallnq/quora/train.csv",
          batch_size=1,
          no_of_irrelevant_samples=4,
          encodingType="NGRAM",
          dense=True)
rcv1TrainingSet = ReutersFileIterator(
          "datasets_smallnq/rcv1/total.json",
          "datasets_smallnq/rcv1/train.json",
          batch_size=1,
          no_of_irrelevant_samples=4,
          encodingType="NGRAM",
          dense=True)

preTrainingSet = RandomBatchIterator(rcv1TrainingSet, quoraTrainingSet)

print("pearsoncoeff of pretraining data sets: {}".format(pearsoncoeff(preTrainingSet, 30000)))


# Calculate correlation within finetuning data set


finetuningSet = NaturalQuestionsFileIterator(
    "/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_datasets/nq/data.csv",
    batch_size=1,
    no_of_irrelevant_samples=1,
    encodingType="NGRAM",
    dense=True
)

avg_pearsonr = 0
print("pearsoncoeff of finetuning data set: {}".format(pearsoncoeff(finetuningSet, 30000)))


# Calculate correlation between pretraining and finetuning data set.

quoraTrainingSet = QuoraFileIterator(
          "datasets_smallnq/quora/total.csv",
          "datasets_smallnq/quora/train.csv",
          batch_size=1,
          no_of_irrelevant_samples=4,
          encodingType="NGRAM",
          dense=True)
rcv1TrainingSet = ReutersFileIterator(
          "datasets_smallnq/rcv1/total.json",
          "datasets_smallnq/rcv1/train.json",
          batch_size=1,
          no_of_irrelevant_samples=4,
          encodingType="NGRAM",
          dense=True)

preTrainingSet = RandomBatchIterator(rcv1TrainingSet, quoraTrainingSet)

finetuningSet = NaturalQuestionsFileIterator(
    "/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_datasets/nq/data.csv",
    batch_size=1,
    no_of_irrelevant_samples=1,
    encodingType="NGRAM",
    dense=True
)

avg_pearsonr = 0
n = 30000
for i in range(n):
    if i % 10000 == 0:
        print("{} pairs looked at".format(i))
    a = preTrainingSet.__next__()
    b = finetuningSet.__next__()
    a_vec = a.get_q_dense() if random.randint(0, 1) else a.get_relevant_dense()
    b_vec = b.get_q_dense() if random.randint(0, 1) else b.get_relevant_dense()
    r, _ = pearsonr(a_vec[0], b_vec[0])
    avg_pearsonr += r / n

print("Cross pearsoncoeff: {}".format(avg_pearsonr))