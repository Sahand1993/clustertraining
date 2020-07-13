import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics import ndcg_score
from tqdm import tqdm

from datasetiterators.fileiterators import WikiQABM25Iterator

bm25TestSetTotal = WikiQABM25Iterator(
    "datasets/wikiqa/test.jsonl",
    "datasets/wikiqa/test.jsonl",
    no_of_irrelevant_samples=4)

corpus = []
ids = []
for example in bm25TestSetTotal:
    corpus.append(example["relevant_tokens"])
    ids.append(example["id"])

bm25 = BM25Okapi(corpus)

bm25TestSetTotal = WikiQABM25Iterator(
    "datasets/wikiqa/test.jsonl",
    "datasets/wikiqa/test.jsonl",
    no_of_irrelevant_samples=4)

ndcg_scores_1 = []
ndcg_scores_3 = []
ndcg_scores_10 = []
ndcg_scores_20 = []
for i, example in enumerate(tqdm(bm25TestSetTotal)):
    query_tokens = example["question_tokens"]
    scores = bm25.get_scores(query_tokens)
    true_scores = np.zeros(len(scores))
    true_scores[i] = 1
    ndcg_scores_1.append(ndcg_score([true_scores], [scores], k=1))
    ndcg_scores_3.append(ndcg_score([true_scores], [scores], k=3))
    ndcg_scores_10.append(ndcg_score([true_scores], [scores], k=10))
    ndcg_scores_20.append(ndcg_score([true_scores], [scores], k=20))

print("BM25 NDCG")
print(sum(ndcg_scores_1) / len(ndcg_scores_1))
print(sum(ndcg_scores_3) / len(ndcg_scores_3))
print(sum(ndcg_scores_10) / len(ndcg_scores_10))
print(sum(ndcg_scores_20) / len(ndcg_scores_20))
