import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv_2d(X, w):
    return tf.nn.conv2d(X, w, padding='SAME', strides=[1, 1, 1, 1])


def max_pool_2X2(X):
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


mnist = input_data.read_data_sets("data/", one_hot=True)
print("Download Done!")

sess = tf.InteractiveSession()

X = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(X, [-1, 28, 28, 1])

w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv_2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2X2(h_conv1)

w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv_2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2X2(h_conv2)

w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entroy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entroy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={X: batch[0], y_: batch[1], keep_prob: 1.0})
        print('第%d次迭代，训练精度为：%g' % (i, train_accuracy))
    train_step.run(feed_dict={X: batch[0], y_: batch[1], keep_prob: 0.5})
print('经过1000次迭代后，在测试集上的精度达到%g' %
      (accuracy.eval(feed_dict={X: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))



