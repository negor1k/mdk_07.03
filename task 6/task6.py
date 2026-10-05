import numpy as np, matplotlib.pyplot as plt

f = lambda x: (x - 3) ** 2
df = lambda x: 2 * (x - 3)
x, lr, path = 10.0, 0.1, []
for i in range(30):
    path.append(x); x = x - lr * df(x)
print(x)
xs = np.linspace(-2, 12, 100); 
plt.plot(xs, f(xs)); 
plt.scatter(path, [f(p) for p in path], c='r'); 
plt.show()


f = lambda x: (x - 3) ** 2
df = lambda x: 2 * (x - 3)
xs = np.linspace(-2, 12, 100)

#шаг3 
plt.figure(figsize=(15, 4)) # Создаем общее широкое окно

# График 1: lr = 0.01
x, lr, path = 10.0, 0.01, []
for i in range(30):
    path.append(x); x = x - lr * df(x)
print(x)
plt.subplot(1, 3, 1) # Разбить на 1 строку и 3 колонки, выбрать 1-ю
plt.plot(xs, f(xs)); plt.scatter(path, [f(p) for p in path], c='r')

# График 2: lr = 0.1
x, lr, path = 10.0, 0.1, []
for i in range(30):
    path.append(x); x = x - lr * df(x)
print(x)
plt.subplot(1, 3, 2) # Выбрать 2-ю колонку
plt.plot(xs, f(xs)); plt.scatter(path, [f(p) for p in path], c='r')

# График 3: lr = 1.1
x, lr, path = 10.0, 1.1, []
for i in range(30):
    path.append(x); x = x - lr * df(x)
print(x)
plt.subplot(1, 3, 3) # Выбрать 3-ю колонку
plt.plot(xs, f(xs)); plt.scatter(path, [f(p) for p in path], c='r')

plt.show()

#шаг 4
rng = np.random.default_rng(0)
X = rng.uniform(0, 5, 50)
y = 2 * X + 1 + rng.normal(0, 0.5, 50)
plt.scatter(X, y); plt.show()

#шаг5
def loss(w, b): return np.mean((w * X + b - y) ** 2)
def grads(w, b):
    p = w * X + b
    return np.mean(2 * (p - y) * X), np.mean(2 * (p - y))

#шаг6
w, b, lr, hist = 0.0, 0.0, 0.05, []
for i in range(200):
    dw, db = grads(w, b); w -= lr * dw; b -= lr * db; hist.append(loss(w, b))
print(w, b)
plt.plot(hist); plt.show()
plt.scatter(X, y); plt.plot(X, w * X + b, 'r'); plt.show()

#шак7
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X.reshape(-1, 1), y)
print("Вес w - ", model.coef_[0])
print("Сдвиг b - ", model.intercept_)
