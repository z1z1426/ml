import random
import numpy as np


def load_dataset(filename):
    data_mat, label_mat = [], []
    with open(filename) as fr:
        lines = fr.readlines()
    for line in lines:
        line_arr = line.strip().split('\t')
        data_mat.append([float(line_arr[0]), float(line_arr[1])])
        label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


# 从0到m中产生一个不为i的整数
def select_jrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


# 使得aj在边界值[L,H]以内
def clip_alpha(aj, h, l):
    if aj > h:
        aj = h
    if l > aj:
        aj = l
    return aj


def smo_simple(data_mat, class_labels, C, toler, max_iter):
    data_matrix, label_mat = np.mat(data_mat), np.mat(class_labels).transpose()
    b = 0
    m, n = np.shape(data_matrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    # alpha_pairs_changed 用来更新的次数
    # 当遍历  连续无更新 maxIter 轮，则认为收敛，迭代结束
    while iter < max_iter:
        alpha_pairs_changed = 0
        for i in range(m):
            fx_i = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[i, :].T)) + b
            e_i = fx_i - float(label_mat[i])
            # toler：容忍错误的程度
            # labelMat[i]*e_i < -toler 则需要alphas[i]增大，但是不能>=C
            # labelMat[i]*e_i > toler 则需要alphas[i]减小，但是不能<=0
            if ((label_mat[i] * e_i < -toler) and (alphas[i] < C)) or \
                    ((label_mat[i] * e_i > toler) and (alphas[i] > 0)):
                j = select_jrand(i, m)
                fx_j = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[j, :].T)) + b
                e_j = fx_j - float(label_mat[j])
                alpha_i_old, alpha_j_old = alphas[i].copy(), alphas[j].copy()
                '''
                计算第二个变量的取值范围
                根据 a1*y1+a2*y2=k 约束条件：
                 1)y1!=y2 : a1-a2=k' -> a2=a1-k' 
                 2)y1==y2 : a1+a2=k' -> a2=-a1+k'
                '''
                if label_mat[i] != label_mat[j]:
                    l = max(0, alphas[j] - alphas[i])
                    h = min(C, C + alphas[j] - alphas[i])
                else:
                    l = max(0, alphas[j] + alphas[i] - C)
                    h = min(C, C + alphas[j] + alphas[i])
                if l == h:
                    print('L==H')
                    continue
                # 计算新的第二个变量值    #计算eta
                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - data_matrix[i, :] * data_matrix[i, :].T \
                    - data_matrix[j, :] * data_matrix[j, :].T
                if eta >= 0:
                    print('eta >= 0')
                    continue
                alphas[j] -= label_mat[j] * (e_i - e_j) / eta
                alphas[j] = clip_alpha(alphas[j], h, l)
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    print('j not moving enough!')
                    continue
                alphas[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])
                b1 = b - e_i - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[i, :].T \
                    - label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[i, :] * data_matrix[j, :].T
                b2 = b - e_j - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[j, :].T \
                     - label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[j, :] * data_matrix[j, :].T
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
                print('第{0}次迭代：i:{1},pairs changed {2}'.format(iter, i, alpha_pairs_changed))
        if alpha_pairs_changed == 0:
            iter += 1
        else:
            iter = 0
        print('iteration number: %d' % iter)
    return b, alphas


if __name__ == '__main__':
    data_arr, label_arr = load_dataset('testSet.txt')
    b, alpha = smo_simple(data_arr, label_arr, 0.6, 0.001, 40)
