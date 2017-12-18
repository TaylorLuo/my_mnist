from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

max_step = 3000  # 最大迭代次数
learning_rate = 0.0013  # 学习率
l2_lambda = 0.00005
dropout = 0.8  # dropout时随机保留神经元的比例
hidden1_input_dim = 800

data_dir = './data'  # 样本数据存储的路径
log_dir = './log'  # 输出日志保存的路径

mnist = input_data.read_data_sets(data_dir, one_hot=True)
sess = tf.InteractiveSession()

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var):
    with tf.name_scope('summaries'):
        # 计算参数的均值
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        # 计算参数的标准差
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))

        # 用直方图记录参数的分布
        tf.summary.histogram('histogram', var)


# input_tensor：特征数据
# input_dim：输入数据的维度大小
# output_dim：输出数据的维度大小(=隐层神经元个数）
# layer_name：命名空间
# act = tf.nn.relu  # 激活函数（默认是relu)
act = 'tf.nn.swish'  # 激活函数（默认是relu)
# act2 = tf.nn.max


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    # 设置命名空间
    with tf.name_scope(layer_name):
        # 调用前面的方法初始化权重w，并调用参数信息记录方法，记录w的信息
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(l2_lambda)(weights))
            variable_summaries(weights)
        # 调用前面的方法初始化偏置项b
        with tf.name_scope('bias'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        # 执行wx+b的线性计算，并用直方图记录下来
        with tf.name_scope('linear_compute'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('linear', preactivate)
        # 将线性输出经过激活函数，并将输出结果也用直方图记录下来

        if(act == 'tf.nn.swish'):
            activations = preactivate*tf.nn.sigmoid(1.7*preactivate, name='activation')
        else:
            activations = act(preactivate, name='activation')

        tf.summary.histogram('activations', activations)

        # 返回激励层的最终输出
        return activations


# 调用隐层创建函数创建一个隐藏层：输入的维度是特征的维度784，神经元个数是800，也就是输出的维度
hidden1 = nn_layer(x, 784, hidden1_input_dim, 'layer1', act = 'tf.nn.swish')
# 创建一个dropout层，随机关闭掉hidden1的一些神经元，并记录keep_prob
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

# 创建一个输出层，输入的维度是上一层的输出:800,输出的维度是分类的类别种类：10，激活函数设置为全等映射identity.（暂且先别使用softmax,会放在之后的损失函数中一起计算）
y = nn_layer(dropped, hidden1_input_dim, 10, 'layer2', act=tf.identity)

# 创建损失函数 使用tf.nn.softmax_cross_entropy_with_logits来计算softmax并计算交叉熵损失,并且求均值作为最终的损失值
with tf.name_scope('loss'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
        l2_loss = tf.add_n(tf.get_collection('losses'))
        cross_entropy = tf.reduce_mean(diff)+l2_loss
tf.summary.scalar('loss', cross_entropy)

# 训练，并计算准确率
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 分别将预测和真实的标签中取出最大值的索引，若相同则返回1，不同则返回0
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        # 求均值即为准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# summaries合并
merged = tf.summary.merge_all()
# 写到指定磁盘路径
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')

# 运行初始化所有变量
tf.global_variables_initializer().run()


def feed_dict(train):
    if train:
        xs, ys = mnist.train.next_batch(100)
        k = dropout
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}


for i in range(max_step):
    if i % 100 == 0:  # 记录测试集的summary和accuracy
        summary, acc, loss = sess.run([merged, accuracy, cross_entropy], feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)
        print('at step %s: Accuracy-->%s  Loss-->%s' % (i, acc, loss))
    else:  # 记录训练集的summary
        if i % 100 == 99:  # 记录执行状态
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged, train_step],
                                  feed_dict=feed_dict(True),
                                  options=run_options,
                                  run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            # print('Adding run metadata for', i)
        else:
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)

train_writer.close()
test_writer.close()
