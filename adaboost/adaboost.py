import numpy as np


def load_simpdata():
    data_mat = np.mat([[1., 2.1],
                      [2., 1.1],
                      [1.3, 1.],
                      [1., 1.],
                      [2., 1.]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, class_labels


def stump_classify(data_matrix, dimen, thresh_val, thresh_ineq):
    # 对数据集每一列的各个特征进行阈值过滤
    ret_array = np.ones((np.shape(data_matrix)[0], 1))
    # 阈值的模式，将小于某一阈值的特征归类为-1
    if thresh_ineq == 'lt':
        ret_array[data_matrix[:, dimen] <= thresh_val] = -1.0
    # 将大于某一阈值的特征归类为-1
    else:
        ret_array[data_matrix[:, dimen] > thresh_val] = -1.0
    return ret_array


def build_stump(data_arr, class_labels, D):
    # 将数据集和标签列表转为矩阵形式
    data_matrix, label_mat = np.mat(data_arr), np.mat(class_labels).T
    m, n = np.shape(data_matrix)
    # 步长或区间总数 最优决策树信息 最优单层决策树预测结果
    num_steps, best_stump, best_classset = 10.0, {}, np.mat(np.zeros((m, 1)))
    # 最小错误率初始化为+∞
    min_error = np.inf
    # 遍历每一列的特征值
    for i in range(n):
        # 找出列中特征值的最小值和最大值
        range_min, range_max = data_matrix[:, i].min(), data_matrix[:, i].max()
        # 求取步长大小或者说区间间隔
        step_size = (range_max - range_min) / num_steps
        # 遍历各个步长区间
        for j in range(-1, int(num_steps) + 1):
            # 两种阈值过滤模式
            for inequal in ['lt', 'gt']:
                # 阈值计算公式：最小值+j(-1<=j<=numSteps+1)*步长
                thresh_val = (range_min + float(j) * step_size)
                # 选定阈值后，调用阈值过滤函数分类预测
                predicted_vals = stump_classify(data_matrix, i, thresh_val, inequal)
                err_arr = np.mat(np.ones((m, 1)))
                # 将错误向量中分类正确项置0
                err_arr[predicted_vals == label_mat] = 0
                # 计算"加权"的错误率
                weighted_error = D.T * err_arr
                # 打印相关信息，可省略
                # print("split: dim %d, thresh %.2f,thresh inequal:\
                #    %s, the weighted error is %.3f",
                #    %(i,threshVal,inequal,weigthedError))
                # 如果当前错误率小于当前最小错误率，将当前错误率作为最小错误率
                # 存储相关信息
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_classset = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    # 返回最佳单层决策树相关信息的字典，最小错误率，决策树预测输出结果
    return best_stump, min_error, best_classset


def adaboost_trainds(data_arr, class_labels, num_it=40):
    weak_class_arr = []
    m = np.shape(data_arr)[0]
    D = np.mat(np.ones((m, 1)) / m)
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(num_it):
        best_stump, error, class_est = build_stump(data_arr, class_labels, D)
        pass


if __name__ == '__main__':
    data_mat, class_labels = load_simpdata()
    D = np.mat(np.ones((5, 1)) / 5)
    build_stump(data_mat, class_labels, D)
    print(1)
