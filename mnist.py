"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

# Import data
data_dir = './data'
mnist = input_data.read_data_sets(data_dir, one_hot=True)

# Create the model
in_units = 784
# 增加一个隐层
h1_units = 800
# 初始化偏置项
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.constant(0.1, shape=[10]))
x = tf.placeholder(tf.float32, [None, in_units])
# 隐层使用relu激活函数
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
# y = tf.matmul(hidden1, W2) + b2
# 用dropout防止过拟合
keep_prob = tf.placeholder("float")
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.identity(tf.matmul(hidden1_drop, W2) + b2)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

# 设置正则化方法 L2正则
regularizer = tf.contrib.layers.l2_regularizer(0.00007)
regularization = regularizer(W1) + regularizer(W2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# 总损失等于交叉熵损失和正则化损失的和
loss = cross_entropy + regularization

train_step = tf.train.AdamOptimizer(0.0013).minimize(loss)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train
print("the accuracy on train datasets:")
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.83})
    if i % 100 == 0:
        print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1}))

# Test trained model
print("the accuracy on test datasets:")
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1}))
