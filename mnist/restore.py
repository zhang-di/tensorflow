import tensorflow as tf
import cv2
img = cv2.imread("d:\\11.png",0)
img.shape = [-1,28*28]

saver = tf.train.import_meta_graph('/model/mnist.ckpt.meta')
# saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "/model/mnist.ckpt")
    soft_max = sess.graph.get_tensor_by_name("output/active4/softmax:0")
    predictions = sess.run(soft_max,{'input/x_input:0':img,'input/drop:0':1})
    for i in predictions:
        print(i)
