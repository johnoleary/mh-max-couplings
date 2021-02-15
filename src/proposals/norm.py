
import random
import numpy as np
import scipy as sp
from numpy.linalg import norm
from scipy import stats as stats

# need:
#   attributes: mode, d
#   methods: log_q_for_ar, log_prop1x_density, prop1x, prop2x

class NormalMaxCoupling(object):
    """A class with all the functions needed to draw from
    1- and d-dimensional maximal couplings with various residual
    treatments. This is the base class for 'Proposal', which is
    more specific to the setting of coupled chains.
    """

    @staticmethod
    def log_p(x, mu, sigma):  # = log N(x ; mu, sigma^2)
        return stats.norm.logpdf(x, loc=mu, scale=sigma)

    @staticmethod
    def norm_pdf(x, mu, sigma):  # = log N(x ; mu, sigma^2)
        return stats.norm.pdf(x, loc=mu, scale=sigma)

    @staticmethod
    def norm_cdf(x, mu, sigma):  # = log N(x ; mu, sigma^2)
        return stats.norm.cdf(x, loc=mu, scale=sigma)

    def _max1d(self, mu0, mu1, sigma):
        """Draw from a coupling of Normal random variables in R^1.
        p0(x) = N(x; mu0, sigma2) and p1(x) = N(x; mu1, sigma2)
        """
        log_p = self.log_p
        draw = stats.norm.rvs(loc=mu0, scale=sigma)
        log_u = np.log(stats.uniform.rvs())
        succeed = (log_p(draw, mu0, sigma) + log_u <= log_p(draw, mu1, sigma))
        if succeed:
            return draw, 'meet'
        else:
            return draw, 'resid'

    def _max1d_parts(self, mu0, mu1, sigma, part):
        assert part in ['meet', 'resid'], 'invalid part option'
        # 'meet' draws from p \propto p0 \wedge p1
        # 'resid' draws from p \propto p0 - p1
        log_p = self.log_p

        accept = False
        while not accept:
            draw = stats.norm.rvs(loc=mu0, scale=sigma)
            log_u = np.log(stats.uniform.rvs())
            under = (log_p(draw, mu0, sigma) + log_u <= log_p(draw, mu1, sigma))
            if (part == 'meet' and under) or (part == 'resid' and not under):
                accept = True
        return draw, part

    def max_indep_1d(self, mu0, mu1, sigma):
        draw0, part0 = self._max1d(mu0, mu1, sigma)
        if part0 == 'meet':
            return (draw0, draw0)
        else:
            draw1, part1 = self._max1d_parts(mu1, mu0, sigma, 'resid')
            return (draw0, draw1)

    def max_refl_1d(self, mu0, mu1, sigma):
        draw0, part0 = self._max1d(mu0, mu1, sigma)
        if part0 == 'meet':
            return (draw0, draw0)
        else:
            draw1 = mu0 + mu1 - draw0
            return (draw0, draw1)

    def _resid_pdf(self, x, mu0, mu1, sigma):
        F0 = lambda x: self.norm_cdf(x, mu0, sigma)
        F1 = lambda x: self.norm_cdf(x, mu1, sigma)
        m = 0.5 * (mu0 + mu1)
        C = np.abs(F0(m) - F1(m))

        p0 = lambda x: self.norm_pdf(x, mu0, sigma)
        p1 = lambda x: self.norm_pdf(x, mu1, sigma)

        return max(0, p0(x) - p1(x)) / C

    def _resid_cdf(self, x, mu0, mu1, sigma):
        F0 = lambda x: self.norm_cdf(x, mu0, sigma)
        F1 = lambda x: self.norm_cdf(x, mu1, sigma)
        m = 0.5 * (mu0 + mu1)
        C = F0(m) - F1(m)
        if mu0 < mu1:
            xwm = min(x, m)
            return (F0(xwm) - F1(xwm)) / C
        else:
            xvm = max(x, m)
            return 1 - (F0(xvm) - F1(xvm)) / C

    def _resid_ppf(self, p, mu0, mu1, sigma):
        assert mu0 != mu1, 'mu0 and mu1 must not be equal'
        m = 0.5 * (mu0 + mu1)
        if mu0 < mu1:
            if p > 1 - 1e-8:
                return m
            else:
                b = [-1e5, m]
        else:
            if p < 1e-8:
                return m
            else:
                b = [m, 1e5]
        F = lambda x: self._resid_cdf(x, mu0, mu1, sigma) - p
        return sp.optimize.root_scalar(F, bracket=b).root

    def max_ot_1d(self, mu0, mu1, sigma):
        draw0, part0 = self._max1d(mu0, mu1, sigma)
        if part0 == 'meet':
            return (draw0, draw0)
        else:
            p = self._resid_cdf(draw0, mu0, mu1, sigma)
            draw1 = self._resid_ppf(p, mu1, mu0, sigma)
            return (draw0, draw1)

    @staticmethod
    def _make_e(x, y):
        """Output y-x unit vec, or a random unit vector if x=y."""
        vec_xy = y - x
        if norm(vec_xy) < 1e-8:
            vec_xy = stats.norm.rvs(size=len(vec_xy))
        e = vec_xy / norm(vec_xy)
        return e

    @staticmethod
    def _proj_out(v, e):
        return v - v.dot(e) * e

    def max_draw(self, mu0, mu1, sigma, resid_mode):
        assert resid_mode in ['indep', 'refl', 'ot', 'semi','refl_full'], 'invalid max coupling residual mode'
        assert len(mu0) == len(mu1), 'incompatible mu lengths'
        _proj_out = self._proj_out

        d = len(mu0)
        e = self._make_e(mu0, mu1)
        mu0_e, mu1_e = mu0.dot(e), mu1.dot(e)
        mu0_orth, mu1_orth = _proj_out(mu0, e), _proj_out(mu1, e)

        # draw e = (y-x)/||y-x|| components
        if resid_mode in ['indep', 'semi']:
            xi_e0, xi_e1 = self.max_indep_1d(mu0_e, mu1_e, sigma)
        elif resid_mode in ['refl', 'refl_full']:
            xi_e0, xi_e1 = self.max_refl_1d(mu0_e, mu1_e, sigma)
        elif resid_mode == 'ot':
            xi_e0, xi_e1 = self.max_ot_1d(mu0_e, mu1_e, sigma)

        # generate proposal x,y
        xi_orth0 = _proj_out(stats.norm.rvs(scale=sigma, size=d), e)
        if resid_mode == 'indep' and not np.isclose(xi_e0, xi_e1):
            xi_orth1 = _proj_out(stats.norm.rvs(scale=sigma, size=d), e)
        elif resid_mode == 'refl_full' and not np.isclose(xi_e0, xi_e1):
            xi_orth1 = -xi_orth0
        else:
            xi_orth1 = xi_orth0

        draw0 = mu0_orth + xi_orth0 + xi_e0*e
        draw1 = mu1_orth + xi_orth1 + xi_e1*e

        return draw0, draw1


class NormalProposal(NormalMaxCoupling):
    def __init__(self, d, sigma2, prop_mode, **kwargs):
        """
        Initialize the Proposal object.

        Mode can be one of the following:
        indep = independent normal draws: X' = X + U, Y' = Y + V
        syn = synchronous coupling: X' = X + U, Y' = Y + U
        refl = reflection coupling: X' = X + e Z1 + Z, Y' = Y - e Z1 + Z
        max_indep = max coupling with independent residuals
        max_refl = max coupling with reflection coupling on residuals
        max_ot = optimal transport
        max_parallel = semi-independent, i.e. indep in e, synchronous in e^\perp
        """

        self.d = d
        self.sigma = np.sqrt(sigma2)
        self.kwargs = kwargs
        self.mode = prop_mode

        # non-maximal couplings: independent, synchronous, and reflection
        if prop_mode == 'indep':
            self.prop2x = self.prop2x_indep
        elif prop_mode == 'syn':
            self.prop2x = self.prop2x_syn
        elif prop_mode == 'refl':
            self.prop2x = self.prop2x_refl
        elif prop_mode == 'semi':
            self.prop2x = self.prop2x_semi

        # simple maximal couplings: indep, reflected, parallel, and OT residuals
        elif prop_mode == 'max_indep':
            self.resid_mode = 'indep'
            self.prop2x = self.prop2x_max
        elif prop_mode == 'max_refl':
            self.resid_mode = 'refl'
            self.prop2x = self.prop2x_max
        elif prop_mode == 'max_refl_full':
            self.resid_mode = 'refl_full'
            self.prop2x = self.prop2x_max
        elif prop_mode == 'max_ot':
            self.resid_mode = 'ot'
            self.prop2x = self.prop2x_max
        elif prop_mode == 'max_semi': #we should rename this to semi
            self.resid_mode = 'semi'
            self.prop2x = self.prop2x_max

        # hybrid couplings
        elif prop_mode == 'hybrid_refl':
            self.resid_mode = 'refl'
            self.hybrid_var = self.kwargs['hybrid_var']
            self.hybrid_cutoff = self.kwargs['hybrid_cutoff']
            self.prop2x = self.prop2x_hybrid

        # full kernel couplings
        elif prop_mode in ('full_max_indep','full_max_refl','full_max_sync'):
            self.prop2x = None

        else:
            raise ValueError('Proposal mode not supported')

    def log_q_for_ar(self, x_from, x_to):
        return 1. #use constant for a symmetric kernel

    def prop1x(self, x_curr):
        x_prop = x_curr + stats.norm.rvs(scale=self.sigma, size=self.d)
        return x_prop

    def log_prop1x_density(self, x_curr, x_next):
        self.Sigma = np.diag(np.ones(self.d) * self.sigma**2)
        lpd = stats.multivariate_normal.logpdf(x_next, mean=x_curr, cov=self.Sigma)
        return lpd

    def prop2x_indep(self, x_curr, y_curr):
        x_prop = x_curr + stats.norm.rvs(scale=self.sigma, size=self.d)
        y_prop = y_curr + stats.norm.rvs(scale=self.sigma, size=self.d)
        return x_prop, y_prop

    def prop2x_syn(self, x_curr, y_curr):
        """Synchronous coupling of Normals."""
        z = stats.norm.rvs(scale=self.sigma, size=self.d)
        x_prop = x_curr + z
        y_prop = y_curr + z
        return x_prop, y_prop

    def prop2x_refl(self, x_curr, y_curr):
        """Reflection coupling of Normals."""
        e = self._make_e(x_curr, y_curr)
        z_orth = self._proj_out(stats.norm.rvs(scale=self.sigma, size=self.d), e)
        z_e = stats.norm.rvs(scale=self.sigma, size=1)
        x_prop = x_curr - e * z_e + z_orth
        y_prop = y_curr + e * z_e + z_orth
        return x_prop, y_prop

    def prop2x_semi(self, x_curr, y_curr):
        """Semi-independent coupling of Normals."""
        e = self._make_e(x_curr, y_curr)
        z_orth = self._proj_out(stats.norm.rvs(scale=self.sigma, size=self.d), e)
        ze_x, ze_y = stats.norm.rvs(scale=self.sigma, size=2)
        x_prop = x_curr - e * ze_x + z_orth
        y_prop = y_curr + e * ze_y + z_orth
        return x_prop, y_prop

    def prop2x_max(self, x_curr, y_curr):
        x_prop, y_prop = self.max_draw(x_curr, y_curr, self.sigma, self.resid_mode)
        return x_prop, y_prop

    def prop2x_hybrid(self, x_curr, y_curr):
        assert self.hybrid_var=='r', 'unsupported cutoff variable'
        assert self.resid_mode=='refl', 'unsupported hybrid type'

        if self.hybrid_var=='r':
            cutoff_curr = norm(y_curr - x_curr)

        if cutoff_curr <= self.hybrid_cutoff:
            return self.prop2x_max(x_curr, y_curr)
        else:
            return self.prop2x_refl(x_curr, y_curr)


class NormalProposalOffset(NormalMaxCoupling):
    def __init__(self, d, sigma2, prop_mode, offset=0, **kwargs):
        """
        Initialize the Proposal object.

        Mode can be one of the following:
        indep = independent normal draws: X' = X + U, Y' = Y + V
        syn = synchronous coupling: X' = X + U, Y' = Y + U
        refl = reflection coupling: X' = X + e Z1 + Z, Y' = Y - e Z1 + Z
        max_indep = max coupling with independent residuals
        max_refl = max coupling with reflection coupling on residuals
        max_ot = optimal transport
        max_parallel = semi-independent, i.e. indep in e, synchronous in e^\perp
        """

        self.d = d
        self.sigma = np.sqrt(sigma2)
        self.kwargs = kwargs
        self.mode = prop_mode
        self.offset = offset

        # non-maximal couplings: independent, synchronous, and reflection
        if prop_mode == 'indep':
            self.prop2x = self.prop2x_indep

        # simple maximal couplings: indep, reflected, parallel, and OT residuals
        elif prop_mode == 'max_indep':
            self.resid_mode = 'indep'
            self.prop2x = self.prop2x_max

        elif prop_mode == 'max_refl':
            self.resid_mode = 'refl'
            self.prop2x = self.prop2x_max

        # full kernel couplings
        elif prop_mode in ('full_max_indep','full_max_refl','full_max_sync'):
            self.prop2x = None

        else:
            raise ValueError('Proposal mode not supported')

    def log_q_for_ar(self, x_from, x_to):
        return self.log_prop1x_density(x_from,x_to)  #use constant for a symmetric kernel

    def log_prop1x_density(self, x_curr, x_next):
        Sigma = np.diag(np.ones(self.d) * self.sigma ** 2)
        lpd = stats.multivariate_normal.logpdf(x_next, mean=x_curr + self.offset, cov=Sigma)
        return lpd

    def prop1x(self, x_curr):
        x_prop = x_curr + self.offset + stats.norm.rvs(scale=self.sigma, size=self.d)
        return x_prop

    def prop2x_indep(self, x_curr, y_curr):
        x_prop = x_curr + self.offset + stats.norm.rvs(scale=self.sigma, size=self.d)
        y_prop = y_curr + self.offset + stats.norm.rvs(scale=self.sigma, size=self.d)
        return x_prop, y_prop

    def prop2x_max(self, x_curr, y_curr):
        x_prop, y_prop = self.max_draw(x_curr + self.offset, y_curr + self.offset,
                                       self.sigma, self.resid_mode)
        return x_prop, y_prop


class MetropGibbsProposal(NormalMaxCoupling):
    def __init__(self, d, sigma2, prop_mode, scan_mode, **kwargs):
        self.d = d
        self.sigma = np.sqrt(sigma2)
        self.kwargs = kwargs
        self.mode = prop_mode
        self.scan_mode = scan_mode
        if scan_mode == 'systematic':
            self.i = 0

        # full kernel couplings
        if prop_mode in ('full_max_indep','full_max_refl','full_max_sync'):
            self.prop2x = None

        # max coupling w/ indep residuals
        elif prop_mode == 'max_indep':
            self.resid_mode = 'indep'
            self.prop2x = self.prop2x_max

        # max coupling w/ refl residuals
        elif prop_mode == 'max_refl':
            self.resid_mode = 'refl'
            self.prop2x = self.prop2x_max

        else:
            raise ValueError('Proposal mode not supported')

    def log_q_for_ar(self, x_from, x_to):
        return 1. #use constant for a symmetric kernel

    def log_prop1x_density(self, x_curr, x_next):
        eps = np.finfo(float).eps

        diff = ~ np.isclose(x_next, x_curr, rtol=1e-08)
        # assert sum(diff) == 1, 'x_curr and x_next differ at != 1 location'
        if sum(diff)==0:
            raise ValueError('sum(diff)==0')
        elif sum(diff)==1:
            x_curr_i, x_next_i = np.squeeze(x_curr[diff]), np.squeeze(x_next[diff])
            return np.log(1/self.d) + stats.norm.logpdf(x_next_i, loc=x_curr_i, scale=self.sigma)
        else: #sum(diff) > 1:
            return np.log(eps)

    def _update_i(self):
        if self.scan_mode=='systematic':
            i = (self.i + 1) % self.d
        elif self.scan_mode=='random':
            i = random.randrange(self.d)
        else:
            raise ValueError("Invalid scan mode")
        self.i = i
        return i

    def prop1x(self, x_curr):
        i = self._update_i()
        x_prop = np.copy(x_curr)
        x_prop[i] = x_curr[i] + stats.norm.rvs(scale=self.sigma)
        return x_prop

    def prop2x_max(self, x_curr, y_curr):
        i = self._update_i()
        x_prop, y_prop = np.copy(x_curr), np.copy(y_curr)
        x_prop[i], y_prop[i] = self.max_draw(x_curr[[i]], y_curr[[i]], self.sigma, self.resid_mode)
        return x_prop, y_prop
