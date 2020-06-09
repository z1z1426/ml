import pandas as pd
import numpy as np
import random


def splitData(data_list, ratio):
    train_size = int(len(data_list)*ratio)
    random.shuffle(data_list)
    train_set = data_list[:train_size]
    test_set = data_list[train_size:]
    return train_set, test_set


def seprate_by_class(dataset):
    class_dict = {}
    p_pre = {}
    for i in dataset:
        label = i[-1]
        if label not in class_dict:
            class_dict[label] = []
        class_dict[label].append(i[:-1])
    for i in class_dict:
        p_pre[i] = len(class_dict[i]) / float(len(dataset))
    return class_dict, p_pre


def mean_and_var(class_list):
    class_list = [float(x) for x in class_list]
    mean = sum(class_list) / (len(class_list))
    var = sum([np.math.pow(x - mean, 2) for x in class_list]) / (len(class_list) - 1)
    return mean, var


def cal_prob(x, mean, var):
    exponent = np.math.exp(np.math.pow(x-mean, 2) / (-2 * var))
    prob = 1 / np.math.sqrt(2 * np.math.pi * var) * exponent
    return prob


def train(train_set):
    class_dict, p_pre = seprate_by_class(train_set)
    summary = {}
    for label, class_list in class_dict.items():
        summary[label] = [mean_and_var(x) for x in zip(*class_list)]
    return summary, p_pre


def cal_class_prob(input_data, attr):
    prob = 1
    for i, item in enumerate(input_data):
        prob *= cal_prob(item, attr[i][0], attr[i][1])
    return prob


def predict_one(input_data, summary, p_pre):
    result = {}
    for label, item in summary.items():
        result[label] = cal_class_prob(input_data, item) * p_pre[label]
    return max(result, key=result.get)


def test(test_set, summary, p_pre):
    correct = 0
    for i in test_set:
        input_data = i[:-1]
        label = i[-1]
        predict = predict_one(input_data, summary, p_pre)
        if predict == label:
            correct += 1
    return correct / len(test_set)


if __name__ == '__main__':
    data_df = pd.read_csv('IrisData.csv')
    data_list = np.array(data_df).tolist()
    train_set, test_set = splitData(data_list, 0.7)
    summary, p_pre = train(train_set)
    print(test(test_set, summary, p_pre))

