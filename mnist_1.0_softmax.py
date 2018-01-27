# -*- coding: UTF-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

tf.set_random_seed(0)
# 下载数据集
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])

# 构造一个Softmax Regression模型
XX = tf.reshape(X, [-1, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
Y = tf.nn.softmax(tf.matmul(XX, W) + b)

# 最小化交叉熵
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

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

    if i % 10 == 0:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))

    if i % 50 == 0:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
        print(str(i) + ": ********* epoch " + str(i * 100 // mnist.train.images.shape[0] + 1) + " ********* ", end='')
        print("test accuracy:" + str(a) + " test loss: " + str(c))

    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})
