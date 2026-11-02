import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np
import copy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
X = np.sort(rng.uniform(0, 1, 30)); y = np.sin(2 * np.pi * X) + rng.normal(0, .3, 30)
Xv = np.sort(rng.uniform(0, 1, 30)); yv = np.sin(2 * np.pi * Xv) + rng.normal(0, .3, 30)

print(f"{'Степень':<10} | {'Train MSE':<12} | {'Val MSE':<12}")
print("-" * 42)
for d in list((1, 4, 15)):
    m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(X[:, None], y)
    train_mse = np.mean((m.predict(X[:, None]) - y)**2)
    val_mse = np.mean((m.predict(Xv[:, None]) - yv)**2)
    print(f"{d:<10} | {train_mse:<12.4f} | {val_mse:<12.4f}")


#2
X_m, y_m = make_moons(n_samples=100, noise=0.3, random_state=42)
Xtr, Xva, ytr, yva = train_test_split(X_m, y_m, test_size=0.5, random_state=42)
Xtr, Xva = torch.FloatTensor(Xtr), torch.FloatTensor(Xva)
ytr, yva = torch.FloatTensor(ytr), torch.FloatTensor(yva)

loss_fn = nn.BCEWithLogitsLoss()

def fit(model, opt, steps=2000):
    tr, va = [], []
    for i in range(steps):
        model.train(); opt.zero_grad(); l = loss_fn(model(Xtr).squeeze(1), ytr); l.backward(); opt.step()
        if i % 50 == 0:
            model.eval()
            with torch.no_grad():
                tr.append(l.item()); va.append(loss_fn(model(Xva).squeeze(1), yva).item())
    return tr, va

big = nn.Sequential(nn.Linear(2,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,1))
tr, va = fit(big, torch.optim.Adam(big.parameters(), 1e-2))

plt.plot(tr, label='train'); plt.plot(va, label='val'); plt.legend(); plt.show()


#3
def fit_early_stopping(model, opt, steps=2000):
    tr, va = [], []
    best_val_loss = float('inf')
    best_weights = None
    patience, patience_counter = 10, 0
    
    for i in range(steps):
        model.train(); opt.zero_grad(); l = loss_fn(model(Xtr).squeeze(1), ytr); l.backward(); opt.step()
        if i % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(Xva).squeeze(1), yva).item()
                tr.append(l.item()); va.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    break
                    
    if best_weights is not None:
        model.load_state_dict(best_weights)
    return tr, va

model_es = nn.Sequential(nn.Linear(2,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,1))
tr_es, va_es = fit_early_stopping(model_es, torch.optim.Adam(model_es.parameters(), 1e-2))


#4
model_wd = nn.Sequential(nn.Linear(2,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,1))
tr_wd, va_wd = fit(model_wd, torch.optim.Adam(model_wd.parameters(), 1e-2, weight_decay=1e-2))


#5
reg = nn.Sequential(
    nn.Linear(2,128), nn.ReLU(), nn.Dropout(.3), 
    nn.Linear(128,128), nn.ReLU(), nn.Dropout(.3), 
    nn.Linear(128,1)
)
tr2, va2 = fit(reg, torch.optim.Adam(reg.parameters(), 1e-2, weight_decay=1e-2))


#6
X_m_large, y_m_large = make_moons(n_samples=2000, noise=0.3, random_state=42)
Xtr_l, Xva_l, ytr_l, yva_l = train_test_split(X_m_large, y_m_large, test_size=0.5, random_state=42)
Xtr_l, Xva_l = torch.FloatTensor(Xtr_l), torch.FloatTensor(Xva_l)
ytr_l, yva_l = torch.FloatTensor(ytr_l), torch.FloatTensor(yva_l)

def fit_large(model, opt, steps=2000):
    tr, va = [], []
    for i in range(steps):
        model.train(); opt.zero_grad(); l = loss_fn(model(Xtr_l).squeeze(1), ytr_l); l.backward(); opt.step()
        if i % 50 == 0:
            model.eval()
            with torch.no_grad():
                tr.append(l.item()); va.append(loss_fn(model(Xva_l).squeeze(1), yva_l).item())
    return tr, va

big_large = nn.Sequential(nn.Linear(2,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,1))
tr_large, va_large = fit_large(big_large, torch.optim.Adam(big_large.parameters(), 1e-2))


#7
def get_metrics(model, X_eval, y_eval, losses_history):
    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(X_eval).squeeze(1)) > 0.5).float()
        acc = (preds == y_eval).float().mean().item()
    return min(losses_history), acc

loss_2, acc_2 = get_metrics(big, Xva, yva, va)
loss_3, acc_3 = get_metrics(model_es, Xva, yva, va_es)
loss_4, acc_4 = get_metrics(model_wd, Xva, yva, va_wd)
loss_5, acc_5 = get_metrics(reg, Xva, yva, va2)
loss_6, acc_6 = get_metrics(big_large, Xva_l, yva_l, va_large)

print(f"{'Регуляризация':<32} | {'Val Loss':<10} | {'Val Accuracy':<12}")
print("-" * 62)
print(f"{'Без регуляризации':<32} | {loss_2:<10.4f} | {acc_2:<12.2%}")
print(f"{'Ранняя остановка':<32} | {loss_3:<10.4f} | {acc_3:<12.2%}")
print(f"{'Weight decay':<32} | {loss_4:<10.4f} | {acc_4:<12.2%}")
print(f"{'Dropout + Weight decay':<32} | {loss_5:<10.4f} | {acc_5:<12.2%}")
print(f"{'Больше данных':<32} | {loss_6:<10.4f} | {acc_6:<12.2%}")