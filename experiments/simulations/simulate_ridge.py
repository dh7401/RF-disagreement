'''
Simulation for ridge regression
'''
import sys 
sys.path.append('../..')
import matplotlib.pyplot as plt

from utils import *


n = 1024 # sample size
d = 512  # input dimension
N_list = list(range(100, 6001, 100))
N_list_asymptotics = list(range(100, 8000, 10))

sigma_ep = 0.5
gamma = 0.01                 
mu = Mu(0.5, 1.5, 1., 5., 1)
phi = d / n

# Assuming ReLU activation
rho_s = 1 / 4
rho_t = 1 / 4
omega_s = mu.get_source_tr() * (1 - 2 / np.pi)
omega_t = mu.get_target_tr() * (1 - 2 / np.pi)
Sigma_s = mu.get_source_cov(d)
Sigma_t = mu.get_target_cov(d)

# Precise asymptotics

I_asymptotics = []
SS_asymptotics = []
SW_asymptotics = []

for N in N_list_asymptotics:
    psi = d / N

    kappa = solve_kappa(phi, psi, gamma, mu)
    tau = (np.sqrt((psi - phi)**2 + 4 * kappa * psi * phi *
            gamma / rho_s) + psi - phi) / 2 / psi / gamma
    tau_bar = 1 / gamma + psi / phi * (tau - 1 / gamma)

    I_disagr_t = 2*rho_t*psi*kappa/(phi*gamma + rho_s*gamma*(tau*psi + tau_bar*phi)*(omega_s + phi*I_s(1, 2, kappa, phi, mu)))\
                    *(gamma*tau*(omega_t + phi*I_t(1, 2, kappa, phi, mu))*I_s(2, 2, kappa, phi, mu) +
                    (sigma_ep**2+I_s(1, 1, kappa, phi, mu))*(omega_s+phi*I_s(1, 2, kappa, phi, mu))*(omega_t + I_t(1, 1, kappa, phi, mu)) +
                    phi/psi*gamma*tau_bar*(sigma_ep**2+phi*I_s(1, 2, kappa, phi, mu))*I_t(2, 2, kappa, phi, mu))
    SS_disagr_t = I_disagr_t - 2*rho_t*kappa**2*(sigma_ep**2+phi*I_s(1, 2, kappa, phi, mu))\
                    *I_t(2, 2, kappa, phi, mu)/rho_s/(1-kappa**2*I_s(2, 2, kappa, phi, mu))
    SW_disagr_t = I_disagr_t - 2*rho_t*psi*kappa**2*(omega_t+phi*I_t(1, 2, kappa, phi, mu))\
                    *I_s(2, 2, kappa, phi, mu)/rho_s/(phi-psi*kappa**2*I_s(2, 2, kappa, phi, mu))
    
    I_asymptotics.append(I_disagr_t)
    SS_asymptotics.append(SS_disagr_t)
    SW_asymptotics.append(SW_disagr_t)

# Simulations

I_simulated = []
SS_simulated = []
SW_simulated = []

num_iter = 100
test_sample_size = 10000

for N in N_list:
    I_list = []
    SS_list = []
    SW_list = []
    for _ in range(num_iter):
        beta = np.random.randn(d, 1)
        
        # SS disagreement
        X = np.sqrt(Sigma_s) @ np.random.randn(d, n)
        Y = X.T @ beta / np.sqrt(d) + sigma_ep * np.random.randn(n, 1)
        W1 = np.random.randn(N, d)
        W2 = np.random.randn(N, d)
        
        F1 = ReLU(W1 @ X / np.sqrt(d))
        F2 = ReLU(W2 @ X / np.sqrt(d))
        K1 = np.linalg.inv(F1.T @ F1 / N + gamma * np.eye(n))
        K2 = np.linalg.inv(F2.T @ F2 / N + gamma * np.eye(n))

        X_test = np.sqrt(Sigma_t) @ np.random.randn(d, test_sample_size)

        yhat1_SS = Y.T @ K1 @ F1.T @ ReLU(W1 @ X_test / np.sqrt(d)) / N
        yhat2_SS = Y.T @ K2 @ F2.T @ ReLU(W2 @ X_test / np.sqrt(d)) / N

        SS_list.append((np.linalg.norm(yhat1_SS - yhat2_SS)**2)/ test_sample_size)

        # I disagreement
        X1 = (np.sqrt(Sigma_s) @ np.random.randn(d, n))
        Y1 = X1.T @ beta / np.sqrt(d) + sigma_ep * np.random.randn(n, 1)
        X2 = (np.sqrt(Sigma_s) @ np.random.randn(d, n))
        Y2 = X2.T @ beta / np.sqrt(d) + sigma_ep * np.random.randn(n, 1)

        F1 = ReLU(W1 @ X1 / np.sqrt(d))
        F2 = ReLU(W2 @ X2 / np.sqrt(d))
        K1 = np.linalg.inv(F1.T @ F1 / N + gamma * np.eye(n))
        K2 = np.linalg.inv(F2.T @ F2 / N + gamma * np.eye(n))

        yhat1_I = Y1.T @ K1 @ F1.T @ ReLU(W1 @ X_test / np.sqrt(d)) / N
        yhat2_I = Y2.T @ K2 @ F2.T @ ReLU(W2 @ X_test / np.sqrt(d)) / N
        
        I_list.append((np.linalg.norm(yhat1_I - yhat2_I)**2)/ test_sample_size)

        # SW disagreement
        X1 = (np.sqrt(Sigma_s) @ np.random.randn(d, n))
        X2 = (np.sqrt(Sigma_s) @ np.random.randn(d, n))
        Y1 = X1.T @ beta / np.sqrt(d) + sigma_ep * np.random.randn(n, 1)
        Y2 = X2.T @ beta / np.sqrt(d) + sigma_ep * np.random.randn(n, 1)

        W = np.random.randn(N, d)
        F1 = ReLU(W @ X1 / np.sqrt(d))
        F2 = ReLU(W @ X2 / np.sqrt(d))
        K1 = np.linalg.inv(F1.T @ F1 / N + gamma * np.eye(n))
        K2 = np.linalg.inv(F2.T @ F2 / N + gamma * np.eye(n))

        X_test = np.sqrt(Sigma_t) @ np.random.randn(d, test_sample_size)

        yhat1_SW = Y1.T @ K1 @ F1.T @ ReLU(W @ X_test / np.sqrt(d)) / N
        yhat2_SW = Y2.T @ K2 @ F2.T @ ReLU(W @ X_test / np.sqrt(d)) / N
        SW_list.append((np.linalg.norm(yhat1_SW - yhat2_SW)**2)/ test_sample_size)

    I_simulated.append(np.mean(np.array(I_list)))
    SS_simulated.append(np.mean(np.array(SS_list)))
    SW_simulated.append(np.mean(np.array(SW_list)))


# Plotting

plt.plot([N/n for N in N_list_asymptotics], I_asymptotics, label='I disagr.')
plt.plot([N/n for N in N_list_asymptotics], SS_asymptotics, label='SS disagr.')
plt.plot([N/n for N in N_list_asymptotics], SW_asymptotics, label='SW disagr.')

plt.scatter([N/n for N in N_list], I_simulated, linewidths=0.6)
plt.scatter([N/n for N in N_list], SS_simulated, linewidths=0.6)
plt.scatter([N/n for N in N_list], SW_simulated, linewidths=0.6)


plt.xlim((0.05, 6.))
plt.xlabel('$\phi/\psi = N/n$')
plt.ylabel('Target disagr.')
plt.legend()
plt.show()
