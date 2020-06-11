from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Tuple, Set
from math import log2
from tqdm import tqdm
from batchiterators.fileiterators import NaturalQuestionsBM25Iterator

LEARNING_RATE = 0.00011702251629896198
CUTOFF_POINTS = [1, 3, 10, 20]


def cosine_sim(a, b, norm_a, norm_b):
    a = np.squeeze(a)
    b = np.squeeze(b)
    return np.dot(a, b) / (norm_a * norm_b)

bm25TestSetTotal = NaturalQuestionsBM25Iterator(
    "datasets_titlenq/nq/test.jsonl",
    no_of_irrelevant_samples=4)

corpus = []
ids = []
for example in bm25TestSetTotal:
    corpus.append(example["relevant_tokens"])
    ids.append(example["id"])

bm25 = BM25Okapi(corpus)

def find_toplist_bm25(query_tokens, limit):
    scores = bm25.get_scores(query_tokens)
    toplist = []
    for _id, score in zip(ids, scores):
        if len(toplist) < limit:
            toplist.append((_id, score))
        else:
            if min(map(lambda x: x[1], toplist)) < score:
                toplist_scores = list(map(lambda tup: tup[1], toplist))
                min_index = toplist_scores.index(min(toplist_scores))
                toplist[min_index] = (_id, score)

    return reversed(sorted(toplist, key=lambda tup: tup[1]))


def ndcg(find_closest_docs):
    # Loop over test set questions
    print("Calculating ndcg scores...")
    failed_by_dssm: Set[int] = set()
    correct_bm25 = 0
    for example in tqdm(bm25TestSetTotal):
        query_id = example["id"]
        closest_docs_ids: List[Tuple] = find_closest_docs(example["question_tokens"], 20)
        try:
            index_of_relevant_doc = list(map(lambda t: t[0], closest_docs_ids)).index(query_id)
        except ValueError:
            index_of_relevant_doc = -1

        if index_of_relevant_doc != 0:
            failed_by_dssm.add(query_id)

        for cutoff_point, ndcg_per_query_at_cutoff in zip(CUTOFF_POINTS, ndcg_per_query):
            if index_of_relevant_doc < cutoff_point and index_of_relevant_doc > -1:
                ndcg_per_query_at_cutoff.append(1 / log2(index_of_relevant_doc + 1 + 1))
            else:
                ndcg_per_query_at_cutoff.append(0)


for i in range(10):
    ndcg_per_query = [] * len(CUTOFF_POINTS)
    for i in range(len(CUTOFF_POINTS)):
        ndcg_per_query.append([])
    bm25TestSetTotal.restart()
    ndcg(find_toplist_bm25)

    # Calculate averages of ndcg over queries
    for i, cutoff_point in enumerate(CUTOFF_POINTS):
        print("ndcg at {}: {}".format(cutoff_point, sum(ndcg_per_query[i]) / len(ndcg_per_query[i])))

