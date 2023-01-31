'''
Experiments on real-world datasets.
'''
import sys 
sys.path.append('../..')
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from load_datasets import *


N_list = np.arange(3000, 50000, 1000)
d = 3072
n = 1000

# Load CIFAR-10-C-Fog (severity 1)
x_s, y_s, x_t, y_t = load_cifar10c(root_dir='data')


shuffle_idx = np.random.permutation(range(x_s.shape[1]))
x_s = x_s[:, shuffle_idx]
y_s = y_s[shuffle_idx, :]
shuffle_idx = np.random.permutation(range(x_t.shape[1]))
x_t = x_t[:, shuffle_idx]
y_t = y_t[shuffle_idx, :]

x_train_s, x_test_s = x_s[:, :n], x_s[:, n:]
y_train_s, y_test_s = y_s[:n, :], y_s[n:, :]
x_test_t = x_t
y_test_t = y_t


I_disagr_s_list = []
I_disagr_t_list = []
SS_disagr_s_list = []
SS_disagr_t_list = []
SW_disagr_s_list = []
SW_disagr_t_list = []
risk_s_list = []
risk_t_list = []
acc_s_list = []
acc_t_list = []
agreement_s_list = []
agreement_t_list = []


for N in N_list:
    print(N)
    X1, X2 = x_train_s[:, :n//2], x_train_s[:, n//2:]
    Y1, Y2 = y_train_s[:n//2, :], y_train_s[n//2:, :]
    W1 = np.random.randn(N, d)
    W2 = np.random.randn(N, d)

    F1 = ReLU(W1@X1/np.sqrt(d))
    F2 = ReLU(W2@X2/np.sqrt(d))
    F3 = ReLU(W1@X2/np.sqrt(d))
    
    K1 = np.linalg.pinv(F1.T@F1/N)
    K2 = np.linalg.pinv(F2.T@F2/N)
    K3 = np.linalg.pinv(F3.T@F3/N)
    
    yhat1_s = Y1.T @ K1 @ F1.T @ ReLU(W1 @ x_test_s / np.sqrt(d)) / N
    yhat2_s = Y2.T @ K2 @ F2.T @ ReLU(W2 @ x_test_s / np.sqrt(d)) / N
    yhat3_s = Y2.T @ K3 @ F3.T @ ReLU(W1 @ x_test_s / np.sqrt(d)) / N
    
    pred1_s = (yhat1_s >= 0.5).astype(float)
    pred2_s = (yhat2_s >= 0.5).astype(float)
    pred3_s = (yhat3_s >= 0.5).astype(float)

    yhat1_t = Y1.T @ K1 @ F1.T @ ReLU(W1 @ x_test_t / np.sqrt(d)) / N
    yhat2_t = Y2.T @ K2 @ F2.T @ ReLU(W2 @ x_test_t / np.sqrt(d)) / N
    yhat3_t = Y2.T @ K3 @ F3.T @ ReLU(W1 @ x_test_t / np.sqrt(d)) / N
    
    pred1_t = (yhat1_t >= 0.5).astype(float)
    pred2_t = (yhat2_t >= 0.5).astype(float)
    pred3_t = (yhat3_t >= 0.5).astype(float)

    I_disagr_s_list.append(np.mean((yhat1_s-yhat2_s)**2))
    I_disagr_t_list.append(np.mean((yhat1_t-yhat2_t)**2))
    SS_disagr_s_list.append(np.mean((yhat2_s-yhat3_s)**2))
    SS_disagr_t_list.append(np.mean((yhat2_t-yhat3_t)**2))
    SW_disagr_s_list.append(np.mean((yhat1_s-yhat3_s)**2))
    SW_disagr_t_list.append(np.mean((yhat1_t-yhat3_t)**2))
    risk_s_list.append(np.mean((y_test_s.T-yhat1_s)**2))
    risk_t_list.append(np.mean((y_test_t.T-yhat1_t)**2))
    agreement_s_list.append(np.mean(pred2_s == pred3_s))
    agreement_t_list.append(np.mean(pred2_t == pred3_t))
    acc_s_list.append(np.mean(y_test_s.T == pred1_s))
    acc_t_list.append(np.mean(y_test_t.T == pred1_t))


# Risk and SS disagreement for CIFAR-10-C-Fog (severity 1)
plt.scatter(risk_s_list, risk_t_list, label='Risk')
plt.scatter(SS_disagr_s_list, SS_disagr_t_list, label='SS disagr.')
plt.xlabel('Source')
plt.ylabel('Target')
plt.legend()
plt.show()
