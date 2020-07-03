from typing import List

import numpy as np
from scipy.linalg import norm
from sklearn.metrics import ndcg_score
import os
from batchiterators.fileiterators import FileIterator, SquadFileIterator, WikiQAFileIterator
from dssm.model_dense_ngram import *

from helpers.helpers import get_model_path

from tqdm import tqdm

LEARNING_RATE = 0.00011702251629896198
optimizer = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE).minimize(logloss)

dssmTestSetTotal = WikiQAFileIterator(
    "datasets/wikiqa/test.csv",
    "datasets/wikiqa/test.csv",
    batch_size=1,
    no_of_irrelevant_samples=4,
    encodingType="NGRAM",
    dense=True,
    shuffle=False)

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



def vectorize_set(dataSet: FileIterator):
    print("Vectorizing test set...")
    dataSet.restart()
    for batch in tqdm(dataSet):
        qId = batch.get_qIds()[0]
        docId = batch.get_docIds()[0]
        qIdToDocId[qId] = docId

        q_vec, doc_vec = sess.run([y_q, y_p], feed_dict = get_feed_dict(batch))

        vecs[qId] = q_vec
        vecs[docId] = doc_vec

        qIds.append(qId)
        docIds.append(docId)

        docIdToIndex[docId] = len(docIds) - 1



def cosine_sim(a, b):
    a = np.squeeze(a)
    b = np.squeeze(b)
    return np.dot(a, b) / (norm(a) * norm(b))


def get_scores(query_vec: np.ndarray):
    scores = map(lambda docId: cosine_sim(query_vec, vecs[docId]), docIds)
    return list(scores)

#initialize model

homeDir = "finetune_pretrained_squad"
for _dir in os.listdir(homeDir):
    if (_dir == ".DS_Store"):
        continue

    vecs = {}
    qIds = []
    docIds = []
    qIdToDocId = {}
    docIdToIndex = {}

    sess = tf.compat.v1.Session()

    #init = tf.compat.v1.global_variables_initializer()
    #sess.run(init)

    modelPath = get_model_path(os.path.join(homeDir, _dir))
    saver = tf.compat.v1.train.Saver(max_to_keep=20)
    saver.restore(sess, modelPath)

    vectorize_set(dssmTestSetTotal)

    ndcg_scores_1 = []
    ndcg_scores_3 = []
    ndcg_scores_10 = []
    ndcg_scores_20 = []
    print("Calculating NDCG...")

    for i, qId in enumerate(tqdm(qIds)):
        q_vec = vecs[qId]

        scores: List[float] = get_scores(q_vec)
        true_scores = np.zeros(len(scores))
        correctDocId = qIdToDocId[qId]
        idxOfCorrect = docIdToIndex[correctDocId]
        true_scores[idxOfCorrect] = 1

        ndcg_scores_1.append(ndcg_score([true_scores], [scores], k=1))
        ndcg_scores_3.append(ndcg_score([true_scores], [scores], k=3))
        ndcg_scores_10.append(ndcg_score([true_scores], [scores], k=10))
        ndcg_scores_20.append(ndcg_score([true_scores], [scores], k=20))

    print("DSSM NDCG:")
    print(sum(ndcg_scores_1) / len(ndcg_scores_1))
    print(sum(ndcg_scores_3) / len(ndcg_scores_3))
    print(sum(ndcg_scores_10) / len(ndcg_scores_10))
    print(sum(ndcg_scores_20) / len(ndcg_scores_20))