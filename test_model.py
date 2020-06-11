# Evaluate model on test set.

from scipy.linalg import norm
import numpy as np
from typing import List, Tuple, Dict, Set
from math import log2
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import pickle

from helpers.helpers import correct_guesses_of_dssm

LEARNING_RATE = 0.00011702251629896198
CUTOFF_POINTS = [1, 3, 10, 20]


def cosine_sim(a, b, norm_a, norm_b):
    a = np.squeeze(a)
    b = np.squeeze(b)
    return np.dot(a, b) / (norm_a * norm_b)


from dssm.model_dense_ngram import *


optimizer = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE).minimize(logloss)

query_vecs: Dict[int, float] = {}  # The vectorized questions as (id, q_vec)
doc_vecs: Dict[int, float] = {}  # The vectorized documents as (id, doc_vec)

# Load test set

from batchiterators.fileiterators import NaturalQuestionsFileIterator, NaturalQuestionsBM25Iterator

dssmTestSetTotal = NaturalQuestionsFileIterator(
    "datasets_titlenq/nq/train.csv",
    batch_size=1,
    no_of_irrelevant_samples=4,
    encodingType="NGRAM",
    dense=True,
    shuffle=False,
    title=True)


def get_BM25_testset(path):
    return NaturalQuestionsBM25Iterator(
        path,
        no_of_irrelevant_samples=0,
        title=True)

testSetBM25Total = get_BM25_testset("datasets_titlenq/nq/test.jsonl")
corpus = [example["relevant_tokens"] for example in testSetBM25Total]
bm25 = BM25Okapi(corpus)

testSetBM25Total = get_BM25_testset("datasets_titlenq/nq/test.jsonl")
query_indices: Dict[int, Dict] = {}
queryIdToTokens: Dict[int, Dict] = {}
for i, example in enumerate(testSetBM25Total):
    query_indices[i] = {"question_tokens": example["question_tokens"], "id": example["id"]}
    queryIdToTokens[example["id"]] = {"question_tokens": example["question_tokens"], "title_tokens": example["relevant_tokens"]}

failed_by_bm25: Set[int] = set()
testSetBM25 = get_BM25_testset("datasets_titlenq/nq/test.jsonl")

def get_feed_dict(batch):
    q_batch = batch.get_q_dense()
    p_batch = batch.get_relevant_dense()
    n1_batch, n2_batch, n3_batch, n4_batch = batch.get_irrelevant_dense()

    feed_dict = {
        x_q: q_batch,
        x_p: p_batch,
        x_n1: n1_batch,
        x_n2: n2_batch,
        x_n3: n3_batch,
        x_n4: n4_batch
    }

    return feed_dict


# Load model

def find_closest_docs_DSSM(query_id, limit):
    cosine_sims = []
    for doc_id, doc_vec in doc_vecs.items():
        sim = cosine_sim(query_vecs[query_id], doc_vec, query_norms[query_id], doc_norms[doc_id])
        cosine_sims.append((doc_id, sim))
    cosine_sims.sort(key=lambda x: x[1])
    return list(reversed(cosine_sims[-limit:]))


modelPaths = [
    "finetune_titlenq/finetune_titlenq_2/model_bs16_lr0.00011702251629896198/tf/dssm-14"
]

saver = tf.compat.v1.train.Saver()
sess = tf.compat.v1.Session()

def test_group_acc(modelPath: str):
    print("\n " + modelPath)
    dssmTestSetTotal.restart()
    saver.restore(sess, modelPath)
    # init = tf.compat.v1.global_variables_initializer()
    # sess.run(init)

    ll_val_overall = 0
    correct_test = 0
    print("Calculating group accuracy...")
    for batch in tqdm(dssmTestSetTotal):
        feed_dict = get_feed_dict(batch)
        (ll_val,) = sess.run([logloss], feed_dict=feed_dict)
        batch_correct = correct_guesses_of_dssm(sess, feed_dict, prob_p, prob_n1, prob_n2, prob_n3, prob_n4)
        correct_test += batch_correct
        ll_val_overall += ll_val

    print(correct_test / dssmTestSetTotal.getNoOfDataPoints())


def vectorize_test_set(modelPath):
    # Vectorize each question and document in the test set
    saver.restore(sess, modelPath)
    print("Vectorizing the test set...")
    f = open("datasets_titlenq/nq/test.csv")
    f.readline()
    for batch in tqdm(dssmTestSetTotal):
        _id = f.readline().split(";")[1]
        query = batch.get_q_dense()
        document = batch.get_relevant_dense()

        query_vec, document_vec = sess.run([y_q, y_p], feed_dict={x_q: query, x_p: document})
        doc_vecs[int(_id)] = document_vec
        #if int(_id) in dssmfailonly:
        query_vecs[int(_id)] = query_vec




#test_group_acc(modelPaths[0])
#testSet.restart()
#dssmfailonly = pickle.load(open("failed_by_dssm_only", "rb"))
vectorize_test_set(modelPaths[0])
query_norms = {query_id: norm(query_vec) for query_id, query_vec in query_vecs.items()}
doc_norms = {doc_id: norm(doc_vec) for doc_id, doc_vec in doc_vecs.items()}

ndcg_per_query = [] * len(CUTOFF_POINTS)
for i in range(len(CUTOFF_POINTS)):
    ndcg_per_query.append([])

def ndcg(find_closest_docs):
    # Loop over test set questions
    print("Calculating ndcg scores...")
    failed_by_dssm: Set[int] = set()
    correct_bm25 = 0
    for (query_id, _), bm25example in zip(tqdm(query_vecs.items()), testSetBM25):

        scores = bm25.get_scores(bm25example["question_tokens"])
        index_of_max = scores.argmax()
        if bm25example["id"] == query_indices[index_of_max]["id"]:
            correct_bm25 += 1
        else:
            failed_by_bm25.add(bm25example["id"])

        closest_docs_ids: List[Tuple] = find_closest_docs(query_id, 20)
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

    print(correct_bm25 / len(testSetBM25))


ndcg(find_closest_docs_DSSM)
# Calculate averages of ndcg over queries
for i, cutoff_point in enumerate(CUTOFF_POINTS):
    print("ndcg at {}: {}".format(cutoff_point, sum(ndcg_per_query[i]) / len(ndcg_per_query[i])))

