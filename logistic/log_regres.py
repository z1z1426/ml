import random

import numpy as np

'''
def load_dataset():
    data_mat, label_mat = [], []
    with open('TestSet.txt') as f:
        fr = f.readlines()
    for line in fr:
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat
'''


def load_dataset():
    with open('TestSet.txt') as fr:
        train = fr.readlines()
    data_mat, label_mat = [], []
    for line in train:
        line = line.strip().split()
        data_mat.append([1.0, float(line[0]), float(line[1])])
        label_mat.append(line[-1])
    return data_mat, label_mat


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def grad_ascent(data_mat, classlabels):
    data_matrix = np.mat(data_mat)
    label_mat = np.mat(classlabels).transpose()
    m, n = np.shape(data_matrix)
    weights = np.ones((n, 1))
    itertimes = 500
    a = 0.01
    for i in range(itertimes):
        error = label_mat - sigmoid(data_matrix * weights)
        weights += a * data_matrix.transpose() * error
    return weights


def stop_grad_ascent0(data_matrix, class_labels):
    m, n = np.shape(data_matrix)
    alpha = 0.01
    weights = np.ones(n)
    for j in range(150):
        for i in range(m):
            h = sigmoid(sum(data_matrix[i] * weights))
            error = class_labels[i] - h
            weights += alpha * error * data_matrix[i]
    return weights


def stop_grad_ascent1(data_matrix, class_labels, num_iter=150):
    m, n = np.shape(data_matrix)
    weights = np.ones(n)
    for j in range(num_iter):
        data_index = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_matrix[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights += alpha * error * data_matrix[rand_index]
            del(data_index[rand_index])
    return weights


def plot_best_fit(weights):
    import matplotlib.pyplot as plt
    # weights = weight.getA()
    data_mat, label_mat = load_dataset()
    data_arr = np.array(data_mat)
    n = np.shape(data_arr)[0]
    x_cord1, y_cord1, x_cord2, y_cord2 = [], [], [], []
    for i in range(n):
        if label_mat[i] == 1:
            x_cord1.append(data_arr[i, 1])
            y_cord1.append(data_arr[i, 2])
        else:
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    ax.scatter(x_cord2, y_cord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (- weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def classify_vector(x, weights):
    prob = sigmoid(sum(x * weights))
    return 1.0 if prob > 0.5 else 0.0


def colic_test():
    with open('HorseColicTraining.txt') as fr:
        train = fr.readlines()
    with open('HorseColicTest.txt') as fr:
        test = fr.readlines()
    training_set, training_labels = [], []
    for line in train:
        current_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(current_line[i]))
        training_set.append(line_arr)
        training_labels.append(float(current_line[21]))
    train_weights = stop_grad_ascent1(np.array(training_set), training_labels, 500)
    error_count, num_test_vec = 0, 0.0
    for line in test:
        num_test_vec += 1.0
        current_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(current_line[i]))
        if int(classify_vector(np.array(line_arr), train_weights)) != int(current_line[21]):
            error_count += 1
    error_rate = (float(error_count) / num_test_vec)
    print("在测试集上的错误率为%f" % error_rate)
    return error_rate


def multi_test():
    num_tests, error_sum = 10, 0.0
    for k in range(num_tests):
        error_sum += colic_test()
    print("在经过%d迭代后平均错误率为%f" % (num_tests, error_sum / float(num_tests)))


if __name__ == '__main__':
    # data_mat, label_mat = load_dataset()
    # weights = stop_grad_ascent1(np.array(data_mat), label_mat)
    # plot_best_fit(weights)
    multi_test()