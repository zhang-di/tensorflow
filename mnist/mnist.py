import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# print mnist.train.images.shape,mnist.train.labels.shape
# (55000, 784) (55000, 10)
# 784 = 28*28
# print mnist.test.images.shape,mnist.test.labels.shape\
# (10000, 784) (10000, 10)
# print mnist.validation.images.shape,mnist.validation.labels.shape
# (5000, 784) (5000, 10)
# img = mnist.train.images[1].reshape([28,28])
# cv2.imshow("img",img)
# cv2.waitKey(0)

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, w) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()
for i in range(5000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 抓取100批数据，训练后w,b的值可以确定,优化
    train_step.run({x:batch_xs,y_:batch_ys})
    if (i+1)%100 == 0:
        train_accuracy = accuracy.eval({x:mnist.train.images,y_:mnist.train.labels})
        print("step %d ,train accuracy is %f" % (i+1,train_accuracy))
    if (i+1)%200 == 0:
        test_accuracy = accuracy.eval({x:mnist.test.images,y_:mnist.test.labels})
        print("=============================step %d ,test accuracy is %f" % (i+1,test_accuracy))
