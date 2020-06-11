from typing import List

import numpy as np
from scipy.linalg import norm
from sklearn.metrics import ndcg_score

from batchiterators.fileiterators import NaturalQuestionsFileIterator
from dssm.model_dense_ngram import *

from tqdm import tqdm

LEARNING_RATE = 0.00011702251629896198
optimizer = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE).minimize(logloss)

modelPath = ""

dssmTestSetTotal = NaturalQuestionsFileIterator(
    "datasets_squad_new/nq/test.csv",
    batch_size=1,
    no_of_irrelevant_samples=4,
    encodingType="NGRAM",
    dense=True,
    shuffle=False,
    title=True)

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



def vectorize_set(dataSet: NaturalQuestionsFileIterator):
    print("Vectorizing test set...")
    dataSet.restart()
    for batch in tqdm(dataSet):
        q_vec, doc_vec = sess.run([y_q, y_p], feed_dict = get_feed_dict(batch))
        query_vecs.append(q_vec)
        doc_vecs.append(doc_vec)


def cosine_sim(a, b):
    a = np.squeeze(a)
    b = np.squeeze(b)
    return np.dot(a, b) / (norm(a) * norm(b))


def get_scores(query_vec: np.ndarray):
    scores = map(lambda doc_vec: cosine_sim(query_vec, doc_vec), doc_vecs)
    return list(scores)

#initialize model

#saver = tf.compat.v1.train.Saver()
#saver.restore(sess, modelPath)
for i in range(9):
    query_vecs = []
    doc_vecs = []
    sess = tf.compat.v1.Session()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    vectorize_set(dssmTestSetTotal)

    ndcg_scores_1 = []
    ndcg_scores_3 = []
    ndcg_scores_10 = []
    ndcg_scores_20 = []
    print("Calculating NDCG...")

    for i, query_vec in enumerate(tqdm(query_vecs)):
        scores: List[float] = get_scores(query_vec)
        true_scores = np.zeros(len(scores))
        true_scores[i] = 1
        ndcg_scores_1.append(ndcg_score([true_scores], [scores], k=1))
        ndcg_scores_3.append(ndcg_score([true_scores], [scores], k=3))
        ndcg_scores_10.append(ndcg_score([true_scores], [scores], k=10))
        ndcg_scores_20.append(ndcg_score([true_scores], [scores], k=20))

    print("DSSM NDCG:")
    print(sum(ndcg_scores_1) / len(ndcg_scores_1))
    print(sum(ndcg_scores_3) / len(ndcg_scores_3))
    print(sum(ndcg_scores_10) / len(ndcg_scores_10))
    print(sum(ndcg_scores_20) / len(ndcg_scores_20))