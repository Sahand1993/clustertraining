import tensorflow as tf
from helpers.helpers import cosine_similarity
import os

NO_OF_TRIGRAMS = 70803

tf.compat.v1.disable_eager_execution()

# tf Graph Input
x_q = tf.compat.v1.placeholder("float32", name="x_q", shape=[None, NO_OF_TRIGRAMS])

x_p = tf.compat.v1.placeholder("float32", name="x_p", shape=[None, NO_OF_TRIGRAMS])

x_n1 = tf.compat.v1.placeholder("float32", name="x_n1", shape=[None, NO_OF_TRIGRAMS])

x_n2 = tf.compat.v1.placeholder("float32", name="x_n2", shape=[None, NO_OF_TRIGRAMS])

x_n3 = tf.compat.v1.placeholder("float32", name="x_n3", shape=[None, NO_OF_TRIGRAMS])

x_n4 = tf.compat.v1.placeholder("float32", name="x_n4", shape=[None, NO_OF_TRIGRAMS])

W_2 = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([NO_OF_TRIGRAMS, 300], mean=0, stddev=0.1, dtype=tf.float32), name="W2")
b_2 = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([1, 300], mean=0, stddev=0.1, dtype=tf.float32), name="b2")

l2_q = tf.compat.v1.tanh(tf.compat.v1.matmul(x_q, W_2) + b_2)
l2_p = tf.compat.v1.tanh(tf.compat.v1.matmul(x_p, W_2) + b_2)
l2_n1 = tf.compat.v1.tanh(tf.compat.v1.matmul(x_n1, W_2) + b_2)
l2_n2 = tf.compat.v1.tanh(tf.compat.v1.matmul(x_n2, W_2) + b_2)
l2_n3 = tf.compat.v1.tanh(tf.compat.v1.matmul(x_n3, W_2) + b_2)
l2_n4 = tf.compat.v1.tanh(tf.compat.v1.matmul(x_n4, W_2) + b_2)

W_3 = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([300, 300], mean=0, stddev=0.1, dtype=tf.float32), name="W3")
b_3 = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([1, 300], mean=0, stddev=0.1, dtype=tf.float32), name="b3")

l3_q = tf.compat.v1.tanh(tf.compat.v1.matmul(l2_q, W_3) + b_3)
l3_p = tf.compat.v1.tanh(tf.compat.v1.matmul(l2_p, W_3) + b_3)
l3_n1 = tf.compat.v1.tanh(tf.compat.v1.matmul(l2_n1, W_3) + b_3)
l3_n2 = tf.compat.v1.tanh(tf.compat.v1.matmul(l2_n2, W_3) + b_3)
l3_n3 = tf.compat.v1.tanh(tf.compat.v1.matmul(l2_n3, W_3) + b_3)
l3_n4 = tf.compat.v1.tanh(tf.compat.v1.matmul(l2_n4, W_3) + b_3)

W_4 = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([300, 128], mean=0, stddev=0.1, dtype=tf.float32), name="W4")
b_4 = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([1, 128], mean=0, stddev=0.1, dtype=tf.float32), name="b4")

y_q = tf.compat.v1.tanh(tf.compat.v1.matmul(l3_q, W_4) + b_4)
y_p = tf.compat.v1.tanh(tf.compat.v1.matmul(l3_p, W_4) + b_4)
y_n1 = tf.compat.v1.tanh(tf.compat.v1.matmul(l3_n1, W_4) + b_4)
y_n2 = tf.compat.v1.tanh(tf.compat.v1.matmul(l3_n2, W_4) + b_4)
y_n3 = tf.compat.v1.tanh(tf.compat.v1.matmul(l3_n3, W_4) + b_4)
y_n4 = tf.compat.v1.tanh(tf.compat.v1.matmul(l3_n4, W_4) + b_4)

r_p = cosine_similarity(y_q, y_p)
r_n1 = cosine_similarity(y_q, y_n1)
r_n2 = cosine_similarity(y_q, y_n2)
r_n3 = cosine_similarity(y_q, y_n3)
r_n4 = cosine_similarity(y_q, y_n4)

sum_r = tf.compat.v1.exp(r_p) + tf.compat.v1.exp(r_n1) + tf.compat.v1.exp(r_n2) + tf.compat.v1.exp(r_n3) + tf.compat.v1.exp(r_n4)

prob_p = tf.compat.v1.math.divide(tf.compat.v1.exp(r_p), sum_r)
prob_n1 = tf.compat.v1.math.divide(tf.compat.v1.exp(r_n1), sum_r)
prob_n2 = tf.compat.v1.math.divide(tf.compat.v1.exp(r_n2), sum_r)
prob_n3 = tf.compat.v1.math.divide(tf.compat.v1.exp(r_n3), sum_r)
prob_n4 = tf.compat.v1.math.divide(tf.compat.v1.exp(r_n4), sum_r)

logloss = -tf.compat.v1.reduce_sum(tf.compat.v1.log(prob_p))
optimizer = tf.compat.v1.train.AdamOptimizer(0.0001).minimize(logloss)
