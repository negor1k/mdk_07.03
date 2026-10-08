import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

X, y = make_blobs(200, centers=2, cluster_std=1.2, random_state=1)
Xt = torch.tensor(X, dtype=torch.float32)
yt = torch.tensor(y, dtype=torch.float32)
w = torch.zeros(2, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
p = torch.sigmoid(Xt @ w + b)
loss = torch.nn.functional.binary_cross_entropy(p, yt)
loss.backward()
print(w.grad, b.grad)


#task8
import torch
import torch.nn as nn
from sklearn.datasets import make_moons



w = torch.zeros(2, requires_grad=True); b = torch.zeros(1, requires_grad=True)
for i in range(500):
    p = torch.sigmoid(Xt @ w + b)
    loss = torch.nn.functional.binary_cross_entropy(p, yt)
    loss.backward()
    with torch.no_grad():
        w -= 0.1 * w.grad; b -= 0.1 * b.grad
    w.grad.zero_(); b.grad.zero_()
print(loss.item())


model = nn.Linear(2, 1)
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for i in range(500):
    opt.zero_grad()
    loss = loss_fn(model(Xt).squeeze(1), yt)
    loss.backward()
    opt.step()
print(loss.item(), model.weight, model.bias)

def train(make_opt, steps=300):
    m = nn.Linear(2, 1); opt = make_opt(m.parameters()); h = []
    for i in range(steps):
        opt.zero_grad(); l = loss_fn(m(Xt).squeeze(1), yt); l.backward(); opt.step(); h.append(l.item())
    return h
plt.plot(train(lambda p: torch.optim.SGD(p, lr=0.1)), label='SGD')
plt.plot(train(lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9)), label='SGD+momentum')
plt.plot(train(lambda p: torch.optim.Adam(p, lr=0.01)), label='Adam')
plt.legend(); plt.show()


Xm, ym = make_moons(300, noise=.2, random_state=0)
Xt = torch.tensor(Xm, dtype=torch.float32); yt = torch.tensor(ym, dtype=torch.float32)
model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for i in range(1000):
    opt.zero_grad(); l = loss_fn(model(Xt).squeeze(1), yt); l.backward(); opt.step()
print('accuracy:', ((model(Xt).squeeze(1) > 0).float() == yt).float().mean().item())