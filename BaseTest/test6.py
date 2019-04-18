# 导入input_data这个类
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 从这个类里调用read_data_sets这个方法
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

# // 为真实的标签添加占位符
y_ = tf.placeholder(tf.float32, [None, 10])

# // 创建交叉熵函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# // 创建会话
sess = tf.InteractiveSession()

# // 初始化所有变量
tf.global_variables_initializer().run()

# //循环1000次训练模型
for _ in range(1000):
    # 获取训练集与标签集，每次获取100个样本
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 喂数据，训练
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
print('训练完成')

# 保存模型
saver = tf.train.Saver()
saver = tf.train.Saver(max_to_keep=1)
saver.save(sess, 'ckpt/mnist.ckpt', global_step=1000)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
