'''
Plot deviation from the line.
'''
import matplotlib.pyplot as plt

from utils import *


if __name__ == '__main__':
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

    # Compute slope in the ridgeless / overparam regime
    kappa = solve_kappa(phi, phi, 0., mu)
    slope = rho_t * (omega_t + I_t(1, 1, kappa, phi, mu)) / \
        rho_s / (omega_s + I_s(1, 1, kappa, phi, mu))

    for gamma in [1e-1, 1e-2, 1e-3, 1e-4]:
        I_disagr_s_list = []
        I_disagr_t_list = []
        SS_disagr_s_list = []
        SS_disagr_t_list = []
        SW_disagr_s_list = []
        SW_disagr_t_list = []

        psi_list = np.arange(0.01, phi, 0.01)
        for psi in psi_list:
            kappa = solve_kappa(phi, psi, gamma, mu)
            tau = (np.sqrt((psi - phi)**2 + 4 * kappa * psi * phi *
                   gamma / rho_s) + psi - phi) / 2 / psi / gamma
            tau_bar = 1 / gamma + psi / phi * (tau - 1 / gamma)

            I_disagr_s = 2*rho_s*psi*kappa/(phi*gamma + rho_s*gamma*(tau*psi + tau_bar*phi)*(omega_s + phi*I_s(1, 2, kappa, phi, mu)))\
                            *(gamma*tau*(omega_s + phi*I_s(1, 2, kappa, phi, mu))*I_s(2, 2, kappa, phi, mu) +
                            (sigma_ep**2+I_s(1, 1, kappa, phi, mu))*(omega_s+phi*I_s(1, 2, kappa, phi, mu))*(omega_s + I_s(1, 1, kappa, phi, mu)) +
                            phi/psi*gamma*tau_bar*(sigma_ep**2+phi*I_s(1, 2, kappa, phi, mu))*I_s(2, 2, kappa, phi, mu))
            SS_disagr_s = I_disagr_s - 2*kappa**2*(sigma_ep**2+phi*I_s(1, 2, kappa, phi, mu))\
                            *I_s(2, 2, kappa, phi, mu)/(1-kappa**2*I_s(2, 2, kappa, phi, mu))
            SW_disagr_s = I_disagr_s - 2*psi*kappa**2*(omega_s+phi*I_s(1, 2, kappa, phi, mu))\
                            *I_s(2, 2, kappa, phi, mu)/(phi-psi*kappa**2*I_s(2, 2, kappa, phi, mu))

            I_disagr_t = 2*rho_t*psi*kappa/(phi*gamma + rho_s*gamma*(tau*psi + tau_bar*phi)*(omega_s + phi*I_s(1, 2, kappa, phi, mu)))\
                            *(gamma*tau*(omega_t + phi*I_t(1, 2, kappa, phi, mu))*I_s(2, 2, kappa, phi, mu) +
                            (sigma_ep**2+I_s(1, 1, kappa, phi, mu))*(omega_s+phi*I_s(1, 2, kappa, phi, mu))*(omega_t + I_t(1, 1, kappa, phi, mu)) +
                            phi/psi*gamma*tau_bar*(sigma_ep**2+phi*I_s(1, 2, kappa, phi, mu))*I_t(2, 2, kappa, phi, mu))
            SS_disagr_t = I_disagr_t - 2*rho_t*kappa**2*(sigma_ep**2+phi*I_s(1, 2, kappa, phi, mu))\
                            *I_t(2, 2, kappa, phi, mu)/rho_s/(1-kappa**2*I_s(2, 2, kappa, phi, mu))
            SW_disagr_t = I_disagr_t - 2*rho_t*psi*kappa**2*(omega_t+phi*I_t(1, 2, kappa, phi, mu))\
                            *I_s(2, 2, kappa, phi, mu)/rho_s/(phi-psi*kappa**2*I_s(2, 2, kappa, phi, mu))

            I_disagr_s_list.append(I_disagr_s)
            I_disagr_t_list.append(I_disagr_t)
            SS_disagr_s_list.append(SS_disagr_s)
            SS_disagr_t_list.append(SS_disagr_t)
            SW_disagr_s_list.append(SW_disagr_s)
            SW_disagr_t_list.append(SW_disagr_t)

        # Plot deviation vs. psi/phi for SS disagreement
        SS_deviation = np.array(SS_disagr_t_list) - \
            slope * np.array(SS_disagr_s_list)
        plt.plot(psi_list / phi, SS_deviation, label=f'$\gamma = {gamma}$')

        plt.xlabel('$\psi / \phi$')
        plt.ylabel('Deviation')
        plt.legend()
        plt.show()
