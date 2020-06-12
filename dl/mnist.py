import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
tf.Session(config=config)

mnist = input_data.read_data_sets('data/', one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples // batch_size


# 参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)  # 平均值
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)  # 标准差
        tf.summary.scalar('max', tf.reduce_max(var))  # 最大值
        tf.summary.scalar('min', tf.reduce_min(var))  # 最小值
        tf.summary.histogram('histogram', var)  # 直方图


# 初始化权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积层
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# 池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')
    with tf.name_scope('x_image'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

with tf.name_scope('conv1'):
    with tf.name_scope('w_conv1'):
        w_conv1 = weight_variable([5, 5, 1, 32])
        variable_summaries(w_conv1)
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([32])
        variable_summaries(b_conv1)
    with tf.name_scope('h_conv1'):
        h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope('conv2'):
    with tf.name_scope('w_conv2'):
        w_conv2 = weight_variable([5, 5, 32, 64])
        variable_summaries(w_conv2)
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([64])
        variable_summaries(b_conv2)
    with tf.name_scope('h_conv2'):
        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    with tf.name_scope('h_pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope('fc1'):
    with tf.name_scope('w_fc1'):
        # 初始化起一个全连接层的权值
        w_fc1 = weight_variable([7 * 7 * 64, 1024])
        variable_summaries(w_fc1)
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1024])
        variable_summaries(b_fc1)
    with tf.name_scope('h_pool2_flat'):
        # 把池化层2的结果扁平化为1维
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64], name='h_pool2_flat')
    with tf.name_scope('h_fc1'):
        # 求第一个全连接层的输出(就是矩阵相乘,和卷积不一样)
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    with tf.name_scope('keep_prob'):
        # dropout
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    with tf.name_scope('h_fc1_drop'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

with tf.name_scope('fc2'):
    with tf.name_scope('w_fc2'):
        # 初始化第二个全连接层的权重偏置
        w_fc2 = weight_variable([1024, 10])
        variable_summaries(w_fc2)
    with tf.name_scope('b_fc2'):
        b_fc2 = bias_variable([10])
        variable_summaries(b_fc2)
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2, name='prediction')

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    trian_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

init = tf.global_variables_initializer()

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 合并所有的summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)

    # 保存图
    train_writer = tf.summary.FileWriter('logging/train', sess.graph)
    test_writer = tf.summary.FileWriter('logging/test', sess.graph)
    for epoch in range(1001):
        # for batch in range(n_batch):
        # 训练模型
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(trian_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})

        # 记录训练集训练的参数
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        # 把所有的参数和训练周期加到write里面去
        train_writer.add_summary(summary, epoch)

        # 记录测试集训练的参数
        batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        test_writer.add_summary(summary, epoch)

        if epoch % 100 == 0:
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images[:10000], y: mnist.train.labels[:10000],
                                                      keep_prob: 1.0})
            print('Iter' + str(epoch) + ',Test Accuracy=' + str(test_acc) + ',Train Accuracy=' + str(train_acc))
