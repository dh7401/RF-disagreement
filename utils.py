'''
Baisc functions.
'''
import numpy as np


class Mu:
    '''
    Class to represent two-mass \mu.
    \mu = r \delta_(eig_1s, eig_1t) + (1 - r) \delta_(eig_2s, eig_2t)
    '''
    def __init__(self, r, eig_1s, eig_2s, eig_1t, eig_2t):
        self.r = r
        self.eig_1s = eig_1s
        self.eig_2s = eig_2s
        self.eig_1t = eig_1t
        self.eig_2t = eig_2t

    def get_source_cov(self, dim):
        '''
        Return d x d source covariance matrix \Sigma_s.
        '''
        dim_1 = int(self.r * dim)
        dim_2 = dim - dim_1
        diag = np.concatenate(
            (self.eig_1s * np.ones(dim_1), self.eig_2s * np.ones(dim_2)))
        return np.diag(diag)

    def get_target_cov(self, dim):
        '''
        Return d x d target covariance matrix \Sigma_t.
        '''
        dim_1 = int(self.r * dim)
        dim_2 = dim - dim_1
        diag = np.concatenate(
            (self.eig_1t * np.ones(dim_1), self.eig_2t * np.ones(dim_2)))
        return np.diag(diag)

    def get_source_tr(self):
        '''
        Return normalized trace of \Sigma_s.
        '''
        return self.r * self.eig_1s + (1 - self.r) * self.eig_2s

    def get_target_tr(self):
        '''
        Return normalized trace of \Sigma_t.
        '''
        return self.r * self.eig_1t + (1 - self.r) * self.eig_2t


def ReLU(x):
    '''
    Return ReLU activation.
    '''
    return np.maximum(0., x)


def I_s(a, b, kappa, phi, mu):
    '''
    Return I_{a, b}(kappa)^s
    '''
    return phi * (mu.r * mu.eig_1s**a / (phi + kappa * mu.eig_1s)**b
                  + (1 - mu.r) * mu.eig_2s**a / (phi + kappa * mu.eig_2s)**b)


def I_t(a, b, kappa, phi, mu):
    '''
    Return I_{a, b}(kappa)^t
    '''
    return phi * (mu.r * mu.eig_1s**(a - 1) * mu.eig_1t / (phi + kappa * mu.eig_1s)**b
                  + (1 - mu.r) * mu.eig_2s**(a - 1) * mu.eig_2t / (phi + kappa * mu.eig_2s)**b)


def solve_kappa(phi, psi, gamma, mu, mode='FP', tolerance=1e-7):
    '''
    Solve self-consistent equation for kappa.
    Mode: 'FP' - fixed point interation
                'GS' - grid search 
    '''
    assert mode in ('FP', 'GS'), 'Invalid mode'

    # Assuming ReLU activation
    rho_s = 0.25
    omega_s = mu.get_source_tr() * (1 - 2 / np.pi)

    if mode == 'FP':
        kappa = 1.
        while True:
            kappa_next = (1 - (np.sqrt((psi - phi)**2 + 4 * kappa * psi * phi * gamma / rho_s) + psi - phi) / 2 / psi)\
                / (omega_s + I_s(1, 1, kappa, phi, mu))
            if abs(kappa_next - kappa) <= tolerance:
                break
            kappa = kappa_next
        return kappa

    if mode == 'GS':
        kappa_grid = np.arange(0, 1 / omega_s, tolerance)
        value_grid = (1 - (np.sqrt((psi - phi)**2 + 4 * kappa_grid * psi * phi * gamma / rho_s) + psi - phi) / 2 / psi)\
            / (omega_s + I_s(1, 1, kappa_grid, phi, mu))
        idx = np.abs(kappa_grid - value_grid).argmin()
        return kappa_grid[idx]
