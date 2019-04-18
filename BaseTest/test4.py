import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

Weights = tf.Variable(tf.random_uniform([1], -0.1, 0.1))
biases = tf.Variable(tf.zeros([1]))
y = Weights * x_data + biases

# 定义损失函数和训练方法
loss = tf.reduce_mean(tf.square(y - y_data))  ##最小化方差
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 启动
sess = tf.Session()
sess.run(init)

# 训练拟合，每一步训练队Weights和biases进行更新
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))  ##每20步输出一下W和b

print(step, sess.run(Weights), sess.run(biases))
