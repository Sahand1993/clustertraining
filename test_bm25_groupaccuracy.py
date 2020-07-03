# Evaluate model on test set.
from rank_bm25 import BM25Okapi
import numpy as np
from tqdm import tqdm


# Load test set

from batchiterators.fileiterators import NaturalQuestionsBM25Iterator

for i in range(10):
    testSet = NaturalQuestionsBM25Iterator(
        "/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_datasets_nqtitles/nq/mid.json",
        "/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_datasets_nqtitles/nq/test.jsonl",
        no_of_irrelevant_samples=4,
        title=False)
    correct_val = 0
    for example in tqdm(testSet):
        corpus = [example["relevant_tokens"]] + example["irrelevant_tokens"]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(example["question_tokens"])
        if max(scores) == 0:
            if np.random.randint(0,5) == 0:
                correct_val += 1
            continue
        if np.argmax(scores) == 0:
            correct_val += 1


    print(correct_val / len(testSet))