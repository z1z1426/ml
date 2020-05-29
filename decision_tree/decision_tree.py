import math
import matplotlib.pyplot as plt


def calc_ShannonEnt(dataset):
    # 计算给定数据集的熵
    m = len(dataset)
    labelcount = {}
    for i in dataset:
        labelcount[i[-1]] = labelcount.get(i[-1], 0) + 1
    entroy = 0.0
    for j in labelcount:
        p = labelcount[j] / float(m)
        entroy -= p * math.log(p, 2)
    return entroy


def create_dataset():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


def split_dataset(dataset, axis, value):
    ret_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset


def choose_bestfeature_to_split(dataset):
    dataset_len = len(dataset)
    best_infogain, best_feature = 0.0, -1
    base_entroy = calc_ShannonEnt(dataset)
    # 计算每个特征的信息增益
    for i in range(len(dataset[0]) - 1):
        feat_unique = set([item[i] for item in dataset])
        new_entroy = 0.0
        for j in feat_unique:
            sub_dataset = split_dataset(dataset, i, j)
            prob = len(sub_dataset) / float(dataset_len)
            new_entroy += prob * calc_ShannonEnt(sub_dataset)
        info_gain = base_entroy - new_entroy
        if info_gain > best_infogain:
            best_infogain, best_feature = info_gain, i
    return best_feature


def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        class_count[vote] = class_count.get(vote, 0) + 1
    sorted_classcount = sorted(class_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_classcount[0][0]


# 创建决策树
def create_tree(dataset, labels):
    class_list = [_[-1] for _ in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(dataset) == 1:
        return majority_cnt(class_list)
    best_feat = choose_bestfeature_to_split(dataset)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    del(labels[best_feat])
    unique_value = set([_[best_feat] for _ in dataset])
    for value in unique_value:
        sub_label = labels[:]
        my_tree[best_feat_label][value] = create_tree(split_dataset(dataset, best_feat, value), sub_label)
    return my_tree


def classify(input_tree, feat_labels, test_vec):
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if isinstance(second_dict[key], dict):
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label


def store_tree(input_tree, filename):
    import pickle
    with open(filename, 'wb') as fw:
        pickle.dump(input_tree, fw)


def grab_tree(filename):
    import pickle
    with open(filename, 'rb') as fr:
        return pickle.load(fr)


if __name__ == '__main__':
    dataset, labels = create_dataset()
    tree_labels = labels[:]
    my_tree = create_tree(dataset, tree_labels)
    store_tree(my_tree, 'tree.txt')
    tree = grab_tree('tree.txt')





