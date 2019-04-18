# 导入input_data这个类
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 从这个类里调用read_data_sets这个方法
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

sess = tf.InteractiveSession()

# // 初始化所有变量
tf.global_variables_initializer().run()

saver = tf.train.Saver(max_to_keep=1)

model_file = tf.train.latest_checkpoint('ckpt/')
saver.restore(sess, model_file)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
