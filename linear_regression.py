import numpy as np


def linear_re():
    lr = 0.001
    x = np.array([[1.0, 2.0, 4.0, 5.0], [1.0, 3.0, 7.0, 9.0], [1.0, 6.0, 10.0, 15.0], [1.0, 4.0, 5.0, 6.0]])
    y = np.array([12.0, 20.0, 32.0, 16.0])
    m = x.shape[0]
    t0 = np.random.random(m).T
    J = np.sum((np.matmul(x, t0)-y)**2)/(2*m)
    for i in range(100000):
        grad0 = np.matmul((np.matmul(x, t0) - y), x) / m
        t0 -= lr * grad0
        e = np.sum((np.matmul(x, t0)-y)**2)/(2*m)
        # e = np.sum((np.matmul(x, t0) + t1 - y) ** 2) / 2
        if abs(J - e) <= 0.000001:
            break
        J = e
    print(t0)
    print(np.matmul(x, t0))
