import numpy as np


def load_dataset(filename):
    with open(filename) as f:
        num_feat = len(f.readline().split('\t')) - 1
        fr = f.readlines()
    data_mat, label_mat = [], []
    for line in fr:
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


def stand_regres(x_arr, y_arr):
    x_mat, y_mat = np.mat(x_arr), np.mat(y_arr).T
    xTx = x_mat.T * x_mat
    if np.linalg.det(xTx) == 0.0:
        print('this matrix is singular,cannot do inverse')
        return
    ws = xTx.I * (x_mat.T * y_mat)
    return ws
