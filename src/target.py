
import random

import numpy as np
from numpy.linalg import norm
from scipy import stats as stats


class NormalTarget(object):
    def __init__(self, d, sigma2=None, Sigma=None, Sigma_inv=None):
        # The target distribution is N(0, I_d \sigma^2) on R^d
        self.d = d
        self.mu = np.zeros(d)

        if sigma2 is None and Sigma is None:
            raise ValueError('Either sigma2 or Sigma must be specified')

        if Sigma is None:
            assert Sigma_inv is None, "Cannot specify Sigma_inv without Sigma"
            self.sigma2 = float(sigma2)
            self.Sigma = np.diag(np.ones(d) * sigma2)
            self.diag_cov = True
        else:
            assert type(Sigma) == np.ndarray, "Sigma must be a np array"
            assert Sigma.shape == (d, d), "Sigma must be of dimension d x d"
            assert sigma2 is None, "cannot specify both sigma2 and Sigma"
            assert Sigma_inv is not None, "for efficiency, must specify both Sigma and Sigma_inv together"
            self.Sigma = Sigma
            self.Sigma_inv = Sigma_inv
            self.diag_cov = False

    def log_pi_for_ar(self, z):
        # drop leading summands that don't depend on z
        if self.diag_cov:
            lpfar = -0.5 * (1 / self.sigma2) * norm(z) ** 2
        else:
            lpfar = -0.5 * np.dot(z, np.dot(self.Sigma_inv, z))
        return lpfar

    def rvs(self, size=1):
        draws = stats.multivariate_normal.rvs(mean=self.mu, cov=self.Sigma, size=size)
        if size == 1:
            draws = np.reshape(draws, (-1))
        return draws


class NormalMixtureTarget(object):
    def __init__(self, mu1=-4, mu2=4, sigma2_target=1):
        # The target distribution is Geom(\rho) on \{0,1,...\}
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma2_target = sigma2_target
        self.sigma_target = np.sqrt(sigma2_target)
        self.d = 1

    def log_pi_for_ar(self, z):
        # drop leading summands that don't depend on z
        pdf1 = stats.norm.pdf(z, loc=self.mu1,scale=self.sigma_target)
        pdf2 = stats.norm.pdf(z, loc=self.mu2, scale=self.sigma_target)
        return np.log(0.5*pdf1+0.5*pdf2)

    def rvs(self, size=1):
        if size == 1:
            wh = random.choices([1, 2], [0.5,0.5])[0]
            if wh==1:
                draws = stats.norm.rvs(loc=self.mu1,scale=self.sigma_target, size=1)
            elif wh==2:
                draws = stats.norm.rvs(loc=self.mu2, scale=self.sigma_target, size=1)
            draws = np.reshape(draws, (-1))
        else:
            raise NotImplementedError('rvs with size > 1 not implemented')
        return draws


class GeomTarget(object):
    def __init__(self, rho):
        # The target distribution is Geom(\rho) on \{0,1,...\}
        self.rho = rho
        self.d = 1

    def log_pi_for_ar(self, z):
        # drop leading summands that don't depend on z
        return stats.geom.logpmf(z, p=self.rho, loc=-1)

    def rvs(self, size=1):
        draws = stats.geom.rvs(p=self.rho, loc=-1, size=size)
        if size == 1:
            draws = np.reshape(draws, (-1))
        return draws


class ExpoTarget(object):
    def __init__(self, scale=1):
        # The target distribution is Expo on [0, \infty)
        self.d = 1
        self.scale = scale

    def log_pi_for_ar(self, z):
        # drop leading summands that don't depend on z
        return stats.expon.logpdf(z,self.scale)

    def rvs(self, size=1):
        draws = stats.expon.rvs(self.scale, size=size)
        if size == 1:
            draws = np.reshape(draws, (-1))
        return draws
