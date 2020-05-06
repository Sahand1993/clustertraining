import tensorflow as tf
import numpy as np

def cosine_similarity(a, b):
    c = tf.compat.v1.sqrt(tf.compat.v1.reduce_sum(tf.compat.v1.multiply(a, a), axis=1))
    d = tf.compat.v1.sqrt(tf.compat.v1.reduce_sum(tf.compat.v1.multiply(b, b), axis=1))
    e = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(a, b), axis=1)
    f = tf.compat.v1.multiply(c, d)
    r = tf.compat.v1.divide(e, f)
    return r


def first_is_largest(f1, f2, f3, f4, f5):
    return np.argmax(np.array([f1, f2, f3, f4, f5])) == 0


def correct_guesses_of_dssm(sess, feed_dict, prob_p, prob_n1, prob_n2, prob_n3, prob_n4):
    p, n1, n2, n3, n4 = sess.run([prob_p, prob_n1, prob_n2, prob_n3, prob_n4], feed_dict=feed_dict)

    m = np.vstack((p, n1, n2, n3, n4))
    guess = np.argmax(m, 0)
    return len(np.where(guess == 0)[0])

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
