from batchiterators.fileiterators import NaturalQuestionsBM25Iterator, NaturalQuestionsFileIterator
from dssm.model_dense_ngram import *
from helpers.helpers import correct_guesses_of_dssm
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import numpy as np

LEARNING_RATE = 0.00011702251629896198
optimizer = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE).minimize(logloss)

testSetDssm = NaturalQuestionsFileIterator(
    "/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_datasets_nqtitles/nq/test.csv",
    batch_size=1,
    no_of_irrelevant_samples=4,
    encodingType="NGRAM",
    dense=True,
    shuffle=False,
    title=True)

for i in range(9):
    testSetBm25 = NaturalQuestionsBM25Iterator(
        "/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_datasets_nqtitles/nq/test.jsonl",
        no_of_irrelevant_samples=4,
        title=True)
    with tf.compat.v1.Session() as sess:
        #init = tf.compat.v1.global_variables_initializer()
        #sess.run(init)

        correct_dssm = 0
        correct_bm25 = 0
        for example_dssm, example_bm25 in tqdm(zip(range(len(testSetBm25)), testSetBm25)):
            #feed_dict = get_feed_dict(example_dssm)
            #batch_correct = correct_guesses_of_dssm(sess, feed_dict, prob_p, prob_n1, prob_n2, prob_n3, prob_n4)
            #correct_dssm += batch_correct

            corpus = [example_bm25["relevant_tokens"]] + example_bm25["irrelevant_tokens"]
            bm25 = BM25Okapi(corpus)
            scores = bm25.get_scores(example_bm25["question_tokens"])
            if max(scores) == 0:
                correct_bm25 += 1 if np.random.rand() < 0.2 else 0
                continue
            if np.argmax(scores) == 0:
                correct_bm25 += 1

        #print(correct_dssm / testSetDssm.getNoOfDataPoints())
        print(correct_bm25 / len(testSetBm25))