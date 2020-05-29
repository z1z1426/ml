import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(test, group, labels, k):
    '''
    :param test: 测试值
    :param group: 训练集
    :param labels: 训练集标签
    :param k: knn中的k值
    :return: 预测标签值
    '''
    distance = ((group - np.tile(test, (group.shape[0], 1))) ** 2).sum(axis=1) / 2
    sort_index = distance.argsort()
    classCount = {}
    for i in range(k):
        classCount[labels[sort_index[i]]] = classCount.get(labels[sort_index[i]], 0) + 1
    classCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    return classCount[0][0]


def file2matrix(filename):
    with open(filename) as fr:
        array_lines = fr.readlines()
    number_of_lines = len(array_lines)
    return_mat = np.zeros((number_of_lines, 3))
    classlabel_vector = []
    index = 0
    for line in array_lines:
        line = line.strip()
        list_fromline = line.split('\t')
        return_mat[index, :] = list_fromline[:3]
        classlabel_vector.append(int(list_fromline[-1]))
        index += 1
    return return_mat, classlabel_vector


def autonorm(dataset):
    minvals = dataset.min(0)
    maxvals = dataset.max(0)
    ranges = maxvals - minvals
    m = dataset.shape[0]
    normdataset = dataset - np.tile(minvals, (m, 1))
    normdataset = normdataset / np.tile(ranges, (m, 1))
    return normdataset, ranges, minvals


def datingClassTest():
    ratio = 0.10
    dating_datamat, dating_labels = file2matrix('data/datingTestSet2.txt')
    # ax = fig.add_subplot(111)
    # ax.scatter(dating_datamat[:, 0], dating_labels[:, 1], 15.0 * np.array(dating_labels), 15.0 * np.array(dating_labels))
    # plt.show()
    norm_mat, ranges, min_vals = autonorm(dating_datamat)
    m = norm_mat.shape[0]
    test_vecs = int(m * ratio)
    error_count = 0.0
    for i in range(test_vecs):
        classifier_result = classify0(norm_mat[i, :], norm_mat[test_vecs:m, :], dating_labels[test_vecs:m], 3)
        print("预测结果为{},真实结果为{}".format(classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1
    print("错误率为%f" % (error_count/float(test_vecs)))


def img2vector(filename):
    return_vect = np.zeros(1024)
    with open(filename) as fr:
        for i in range(32):
            line_str = fr.readline()
            for j in range(32):
                return_vect[32 * i + j] = int(line_str[j])
    return return_vect


def handwriting_classtest():
    hwlabels = []
    training_filelist = os.listdir('data/trainingDigits')
    m = len(training_filelist)
    training_mat = np.zeros((m, 1024))
    for i in range(m):
        filename_str = training_filelist[i]
        classnum_str = int(filename_str.split('_')[0])
        hwlabels.append(classnum_str)
        training_mat[i] = img2vector('data/trainingDigits/{}'.format(filename_str))
    test_filelist = os.listdir('data/testDigits')
    m = len(test_filelist)
    error_count = 0
    for i in range(m):
        filename_str = test_filelist[i]
        testnum_str = int(filename_str.split('_')[0])
        test_mat = img2vector('data/testDigits/{}'.format(filename_str))
        pred_label = classify0(test_mat, training_mat, hwlabels, 3)
        # print("预测结果为{},真实结果为{}".format(pred_label, label))
        if pred_label != testnum_str:
            error_count += 1
    print("错误率为%f" % (error_count / float(m)))


if __name__ == '__main__':
    handwriting_classtest()
