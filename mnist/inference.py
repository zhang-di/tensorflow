import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from  PIL import Image
import cv2
img = cv2.imread("d:\\11.png",0)
img.shape = [-1,28*28]
# print(img.shape)

import os
import time
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

start = time.time()

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

# print mnist.train.images.shape,mnist.train.labels.shape
# (55000, 784) (55000, 10)
# 784 = 28*28
# print mnist.test.images.shape,mnist.test.labels.shape\
# (10000, 784) (10000, 10)
# print mnist.validation.images.shape,mnist.validation.labels.shape
# (5000, 784) (5000, 10)
with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32, [None, 784], name='x_input')
    x_image = tf.reshape(xs, [-1, 28, 28, 1])
    ys = tf.placeholder(tf.float32, [None, 10], name='y_input')
    keep_prob = tf.placeholder(tf.float32, name='drop')
with tf.name_scope('hidden'):
    with tf.name_scope('conv32'):
        with tf.name_scope('weight1'):
            W_1 = tf.Variable(tf.random_normal([5, 5, 1, 32], stddev=0.1), name='conv32_W')
            tf.summary.histogram('conv32/W_1', W_1)
        with tf.name_scope('bias1'):
            b_1 = tf.Variable(tf.constant(0.1, tf.float32, [32]), name='conv32_b')
            tf.summary.histogram('conv32/b_1', b_1)
        with tf.name_scope('active1'):
            h_conv32 = tf.nn.relu(tf.nn.conv2d(x_image, W_1, strides=[1, 1, 1, 1], padding='SAME') + b_1, name='relu')
    with tf.name_scope("pool1"):
        h_pool1 = tf.nn.max_pool(h_conv32, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name='pool')
    # -1 14 14 32
    with tf.name_scope('conv64'):
        with tf.name_scope('weight2'):
            W_2 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.1), name='conv64_W')
            tf.summary.histogram('conv64/W_2', W_2)
        with tf.name_scope('bias2'):
            b_2 = tf.Variable(tf.constant(0.1, tf.float32, [64]), name='conv64_b')
            tf.summary.histogram('conv64/b_2', b_2)
        with tf.name_scope('active2'):
            h_conv64 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_2, strides=[1, 1, 1, 1], padding='SAME') + b_2, name='relu')
    with tf.name_scope("pool2"):
        h_pool2 = tf.nn.max_pool(h_conv64, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name='pool')
# -1 7 7 64
    with tf.name_scope("fc1"):
        h_pool2_flatten = tf.reshape(h_pool2, [-1, 7*7*64])
        with tf.name_scope('weight3'):
            W_3 = tf.Variable(tf.random_normal([7*7*64, 1024], stddev=0.1), name='fc1_W')
            tf.summary.histogram('fc1/W_3', W_3)
        with tf.name_scope('bias3'):
            b_3 = tf.Variable(tf.constant(0.1, tf.float32, [1024]), name='fc1_b')
            tf.summary.histogram('fc1/b_3', b_3)
        with tf.name_scope('active3'):
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flatten, W_3) + b_3, name='relu')
        with tf.name_scope('dropout'):
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)
with tf.name_scope("output"):
    with tf.name_scope('weight4'):
        W_4 = tf.Variable(tf.random_normal([1024, 10], stddev=0.1), name='fc2_W')
        tf.summary.histogram('fc2/W_4', W_4)
    with tf.name_scope('bias4'):
        b_4 = tf.Variable(tf.constant(0.1, tf.float32, [10]), name='fc2_b')
        tf.summary.histogram('fc2/b_4', b_4)
    with tf.name_scope('active4'):
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_4) + b_4, name='softmax')
with tf.name_scope('loss'):
    loss = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(y_conv), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    # train = tf.train.AdamOptimizer(1e-4).minimize(loss)
    train = tf.train.AdamOptimizer(1e-4).minimize(loss)
# accuracy
correct_prediction = tf.equal(tf.argmax(ys, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(init)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs', sess.graph)

    for i in range(1001):
        x_batch, y_batch = mnist.train.next_batch(50)
        sess.run(train, feed_dict={xs:x_batch, ys:y_batch, keep_prob:0.5})
        if i%100 == 0:

            result = sess.run(merged, feed_dict={xs: x_batch, ys: y_batch, keep_prob: 1})
            writer.add_summary(result, i)
            print(i, ' step train ', sess.run(accuracy, feed_dict={xs: x_batch, ys: y_batch, keep_prob: 1}))
    x_test, y_test = mnist.test.next_batch(10)
    print(sess.run(tf.argmax(sess.run(y_conv, feed_dict={xs: img, keep_prob: 1}),1)))
end = time.time()
print("function time is : ", end-start)









