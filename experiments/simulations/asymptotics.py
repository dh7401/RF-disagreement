'''
Plot exact asymptotics of disagreements and risk.
'''
import matplotlib.pyplot as plt

from helper.utils import *


if __name__ == '__main__':
    gamma = 0
    sigma_ep = 0.01
    mu = Mu(0.4, 0.1, 1, 1, 0.1)
    phi = 0.5

    # Assuming ReLU activation
    rho_s = 0.25
    rho_t = 0.25
    m_s = mu.get_source_tr()
    m_t = mu.get_target_tr()
    omega_s = m_s * (1 - 2 / np.pi)
    omega_t = m_t * (1 - 2 / np.pi)

    I_disagr_s_list = []
    I_disagr_t_list = []
    SS_disagr_s_list = []
    SS_disagr_t_list = []
    SW_disagr_s_list = []
    SW_disagr_t_list = []
    risk_s_list = []
    risk_t_list = []

    psi_list = np.arange(0.01, 1000, 0.01)
    for psi in psi_list:
        if phi == psi:
            continue
        kappa = solve_kappa(phi, psi, gamma, mu)

        I_disagr_s = 2*psi*kappa*(sigma_ep**2+I_s(1, 1, kappa, phi, mu))\
                        *(omega_s+I_s(1, 1, kappa, phi, mu))/abs(phi-psi)
        if phi > psi:
            I_disagr_s += 2*kappa*(sigma_ep**2+phi*I_s(1, 2, kappa, phi ,mu))*I_s(2, 2, kappa, phi, mu)\
                        /(omega_s+phi*I_s(1, 2, kappa, phi ,mu))
        else:
            I_disagr_s += 2*kappa*(omega_s+phi*I_s(1, 2, kappa, phi ,mu))*I_s(2, 2, kappa, phi, mu)\
                        /(omega_s+phi*I_s(1, 2, kappa, phi ,mu))

        SS_disagr_s = I_disagr_s - 2*kappa**2*(sigma_ep**2+phi*I_s(1, 2, kappa, phi, mu))*I_s(2, 2, kappa, phi, mu)\
                        /(1-kappa**2*I_s(2, 2, kappa, phi, mu))
        SW_disagr_s = I_disagr_s - 2*psi*kappa**2*(omega_s+phi*I_s(1, 2, kappa, phi, mu))*I_s(2, 2, kappa, phi, mu)\
                        /(phi-psi*kappa**2*I_s(2, 2, kappa, phi, mu))
        risk_s = 0.5*I_disagr_s + phi*I_s(1, 2, kappa, phi, mu)


        I_disagr_t = 2*rho_t*psi*kappa*(sigma_ep**2+I_s(1, 1, kappa, phi, mu))*(omega_t+I_t(1, 1, kappa, phi, mu))\
                        /rho_s/abs(phi-psi)
        if phi > psi:
            I_disagr_t += 2*rho_t*kappa*(sigma_ep**2+phi*I_s(1, 2, kappa, phi ,mu))*I_t(2, 2, kappa, phi, mu)\
                        /rho_s/(omega_s+phi*I_s(1, 2, kappa, phi ,mu))
        else:
            I_disagr_t += 2*rho_t*kappa*(omega_t+phi*I_t(1, 2, kappa, phi ,mu))*I_s(2, 2, kappa, phi, mu)/rho_s\
                        /(omega_s+phi*I_s(1, 2, kappa, phi ,mu))

        SS_disagr_t = I_disagr_t - 2*rho_t*kappa**2*(sigma_ep**2+phi*I_s(1, 2, kappa, phi, mu))\
                        *I_t(2, 2, kappa, phi, mu)/rho_s/(1-kappa**2*I_s(2, 2, kappa, phi, mu))
        SW_disagr_t = I_disagr_t - 2*rho_t*psi*kappa**2*(omega_t+phi*I_t(1, 2, kappa, phi, mu))\
                        *I_s(2, 2, kappa, phi, mu)/rho_s/(phi-psi*kappa**2*I_s(2, 2, kappa, phi, mu))
        risk_t = 0.5*I_disagr_t + (1 - np.sqrt(rho_t/rho_s))**2 * m_s + \
                        2*(1-np.sqrt(rho_t/rho_s))*np.sqrt(rho_t/rho_s)*I_t(1, 1, kappa, phi, mu) + \
                        phi*rho_t*I_t(1, 2, kappa, phi, mu)/rho_s

        I_disagr_s_list.append(I_disagr_s)
        I_disagr_t_list.append(I_disagr_t)
        SS_disagr_s_list.append(SS_disagr_s)
        SS_disagr_t_list.append(SS_disagr_t)
        SW_disagr_s_list.append(SW_disagr_s)
        SW_disagr_t_list.append(SW_disagr_t)
        risk_s_list.append(risk_s)
        risk_t_list.append(risk_t)


    # Plot target vs. source disagreements in underparametrized regime
    threshold_idx = np.where(psi_list == 0.5)[0][0]
    plt.plot(I_disagr_s_list[50:], I_disagr_t_list[50:], label='I')
    plt.plot(SS_disagr_s_list[50:], SS_disagr_t_list[50:], label='SS')
    plt.plot(SW_disagr_s_list[50:], SW_disagr_t_list[50:], label='SW')
    plt.xlim((0, 1.5))
    plt.ylim((0, 1.5))

    plt.xlabel('Source disagr.')
    plt.ylabel('Target disagr.')
    plt.legend()
    plt.show()
