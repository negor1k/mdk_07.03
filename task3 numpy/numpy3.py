import numpy as np
#step1
a = np.array([3, 1, 4, 1, 5])
print('step1')
print(a * 2, a + 10, a ** 2)
print(a.mean(), a.min(), a.max(), a.sum())

#step2
M = np.arange(12).reshape(3, 4)
print('step2')
print(M, M.shape)
print('строка 0:', M[0]); print('столбец 1:', M[:, 1]); print('элемент:', M[2, 3])
print(M.T.shape)

#step3
print('step3')
print(M.mean(axis=0))  # 4 числа
print(M.mean(axis=1))  # 3 числа

#step4
A = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
w = np.array([[1], [0], [-1]])          # (3, 1)
print('step4')
print(A @ w)                            # (2, 1)
print(A + np.array([10, 20, 30]))

#step5
rng = np.random.default_rng(0)
X = rng.normal(size=(100, 5))
w = np.array([0.5, -1.0, 2.0, 0.0, 1.5]); b = 0.3
y = X @ w + b
print('step5')
print(X.shape, w.shape, y.shape, y[:5])

