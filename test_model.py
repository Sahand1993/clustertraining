# Evaluate model on test set.

from scipy.linalg import norm
import numpy as np
from typing import List, Tuple, Dict
from math import log2
from tqdm import tqdm

from helpers.helpers import correct_guesses_of_dssm

#LEARNING_RATE = 0.00011702251629896198
LEARNING_RATE = 0.0008497018999399376
CUTOFF_POINTS = [1, 3, 10 ,20]

def cosine_sim(a, b, norm_a, norm_b):
    a = np.squeeze(a)
    b = np.squeeze(b)
    return np.dot(a, b) / (norm_a * norm_b)

from dssm.model_dense_ngram import *
optimizer = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE).minimize(logloss)

queries: Dict[int, float] = {} # The vectorized questions as (id, q_vec)
docs: Dict[int, float] = {} # The vectorized documents as (id, doc_vec)

# Load test set

from batchiterators.fileiterators import NaturalQuestionsFileIterator
testSet = NaturalQuestionsFileIterator(
    "datasets_smallnq/nq/test.csv",
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

# Load model

def find_closest_docs(query_id, limit):
    cosine_sims = []
    for doc_id, doc_vec in docs.items():
        sim = cosine_sim(queries[query_id], doc_vec, query_norms[query_id], doc_norms[doc_id])
        cosine_sims.append((doc_id, sim))
    cosine_sims.sort(key=lambda x: x[1])
    return cosine_sims[-limit:]

modelPath = "hyperparams_smallnq_ngram/run1/model_bs32_lr0.0008497018999399376/tf/dssm-8"
saver = tf.compat.v1.train.Saver()
with tf.compat.v1.Session() as sess:
    saver.restore(sess, modelPath)
    #init = tf.compat.v1.global_variables_initializer()
    #sess.run(init)

    ll_val_overall = 0
    correct_val = 0
    # for batch in tqdm(testSet):
    #     feed_dict = get_feed_dict(batch)
    #     (ll_val,) = sess.run([logloss], feed_dict=feed_dict)
    #     batch_correct = correct_guesses_of_dssm(sess, feed_dict, prob_p, prob_n1, prob_n2, prob_n3, prob_n4)
    #     correct_val += batch_correct
    #     ll_val_overall += ll_val

    #print(correct_val / testSet.getNoOfDataPoints())
    # Vectorize each question and document in the test set
    for i, batch in enumerate(testSet):
        query = batch.get_q_dense()
        document = batch.get_relevant_dense()

        query_vec, document_vec = sess.run([y_q, y_p], feed_dict={x_q:query, x_p:document})
        queries[i] = query_vec
        docs[i] = document_vec

        if i % 1000 == 0:
            print("Converted {} queries and {} documents to vectors.".format(i, i))

query_norms = {query_id : norm(query_vec) for query_id, query_vec in queries.items()}
doc_norms = {doc_id : norm(doc_vec) for doc_id, doc_vec in docs.items()}

ndcg_per_query = [[]] * len(CUTOFF_POINTS)
def ndcg():
    # Loop over test set questions
    for query_id, _ in queries.items():
        closest_docs_ids: List[Tuple] = find_closest_docs(query_id, 20)
        try:
            index_of_relevant_doc = closest_docs_ids.index(query_id)
        except ValueError:
            index_of_relevant_doc = -1

        for cutoff_point, ndcg_per_query_at_cutoff in zip(CUTOFF_POINTS, ndcg_per_query):
            if index_of_relevant_doc < cutoff_point and index_of_relevant_doc > -1:
                ndcg_per_query_at_cutoff.append(1 / log2(index_of_relevant_doc + 1 + 1))
            else:
                ndcg_per_query_at_cutoff.append(0)

        if query_id % 100 == 0:
            print("calculated ndcg for {} queries".format(query_id))

ndcg()
# Calculate averages of ndcg over queries
for i, cutoff_point in enumerate(CUTOFF_POINTS):
    print("ndcg at {}: {}".format(cutoff_point, sum(ndcg_per_query[i]) / len(ndcg_per_query[i])))

