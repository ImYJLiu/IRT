import numpy as np
import numpy.linalg as LA

SIGMA = 6
EPS = 0.0001


# 生成方差相同, 均值不同的样本
def generate_data():
    mu1 = 20
    mu2 = 40
    N = 1000
    X = np.zeros(N)
    for i in range(N):
        temp = np.random.uniform(0, 1)  # Z ~ U[0,1]
        if temp > 0.5:
            X[i] = temp * SIGMA + mu1  # X1 = Z * 6 + 20
        else:
            X[i] = temp * SIGMA + mu2  # X2 = Z * 6 + 40
    return X


def my_EM(X):
    N = X.shape[0]
    k = 2
    mu = np.random.rand(k)
    Posterior = np.zeros((N, 2))
    # 先求后验概率
    for _ in range(1000):
        for i in range(N):
            p = np.exp(-1 / (2 * SIGMA ** 2) * (X[i] - mu) ** 2)
            print(p.shape)
            Posterior[i, :] = p / np.sum(p)
        oldmu = mu.copy()
        # 最大化
        numerator = np.dot(X, Posterior)
        dominator = Posterior.sum(axis=0)
        mu = numerator / dominator
        if LA.norm(mu - oldmu) < EPS:
            return mu


if __name__ == '__main__':
    X = generate_data()
    # print(X)
    my_EM(X)
