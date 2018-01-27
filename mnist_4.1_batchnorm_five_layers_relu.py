# -*- coding: UTF-8 -*-

import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

tf.set_random_seed(0)
# 下载数据集
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])
lr = tf.placeholder(tf.float32)
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)
L = 200
M = 100
N = 60
P = 30
Q = 10
# truncated_normal：截断正态分布
W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1))  # 784 = 28 * 28
B1 = tf.Variable(tf.ones([L]) / 10)
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.ones([M]) / 10)
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B3 = tf.Variable(tf.ones([N]) / 10)
W4 = tf.Variable(tf.truncated_normal([N, P], stddev=0.1))
B4 = tf.Variable(tf.ones([P]) / 10)
W5 = tf.Variable(tf.truncated_normal([P, Q], stddev=0.1))
B5 = tf.Variable(tf.ones([Q]) / 10)


def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    # adding the iteration prevents from averaging across non-existing iterations
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages


def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    return Ylogits, tf.no_op()


# 构建模型
XX = tf.reshape(X, [-1, 784])
Y1l = tf.matmul(XX, W1)
Y1bn, update_ema1 = batchnorm(Y1l, tst, iter, B1)
Y1 = tf.nn.relu(Y1bn)
Y2l = tf.matmul(Y1, W2)
Y2bn, update_ema2 = batchnorm(Y2l, tst, iter, B2)
Y2 = tf.nn.relu(Y2bn)
Y3l = tf.matmul(Y2, W3)
Y3bn, update_ema3 = batchnorm(Y3l, tst, iter, B3)
Y3 = tf.nn.relu(Y3bn)
Y4l = tf.matmul(Y3, W4)
Y4bn, update_ema4 = batchnorm(Y4l, tst, iter, B4)
Y4 = tf.nn.relu(Y4bn)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)
update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)

# 最小化交叉熵
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * 100
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# 正确率
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 初始化变量
init = tf.global_variables_initializer()
# 启动图 (graph)
sess = tf.Session()
sess.run(init)

# 开始训练模型，这里我们让模型循环训练1001次
for i in range(1001):
    batch_X, batch_Y = mnist.train.next_batch(100)

    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 1000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i / decay_speed)

    if i % 10 == 0:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y, tst: False})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")

    if i % 50 == 0:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels, tst: True})
        print(str(i) + ": ********* epoch " + str(i * 100 // mnist.train.images.shape[0] + 1) + " ********* ", end='')
        print("test accuracy:" + str(a) + " test loss: " + str(c))

    sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate, tst: False})
    sess.run(update_ema, {X: batch_X, Y_: batch_Y, tst: False, iter: i})
