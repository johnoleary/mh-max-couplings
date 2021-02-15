
import random
import numpy as np
import scipy as sp
from numpy.linalg import norm
from scipy import stats as stats

class RWProposal(object):
    def __init__(self, gamma, prop_mode, **kwargs):
        self.gamma = gamma
        assert not np.isclose(gamma,0), "We require gamma > 0"
        assert not np.isclose(gamma, 1), "We require gamma < 1"

        self.d = 1
        self.kwargs = kwargs
        self.mode = prop_mode

        # non-maximal couplings: independent, synchronous, and reflection
        if prop_mode == 'indep':
            self.prop2x = self.prop2x_indep

        elif prop_mode == 'syn':
            self.prop2x = self.prop2x_syn

        # to be implemented: max_indep, max_refl, max_sync

        # full kernel couplings
        elif prop_mode in ('full_max_indep','full_max_refl','full_max_sync'):
            self.prop2x = None

        else:
            raise ValueError(f'Proposal mode {prop_mode} not supported')

    def log_q_for_ar(self, x_from, x_to):
        if np.isclose(x_to, x_from + 1):
            return np.log(self.gamma)
        elif np.isclose(x_to, x_from - 1):
            return np.log(1-self.gamma)
        else:
            eps = np.finfo(float).eps
            return np.log(eps)

    def log_prop1x_density(self, x_curr, x_next):
        return self.log_q_for_ar(x_from, x_to)

    def prop1x(self, x_curr):
        gamma = self.gamma
        x_prop = x_curr + random.choices([-1,1],[gamma,1-gamma])[0]
        return x_prop

    def prop2x_indep(self, x_curr, y_curr):
        gamma = self.gamma
        x_prop = x_curr + random.choices([-1,1],[gamma,1-gamma])[0]
        y_prop = y_curr + random.choices([-1,1],[gamma,1-gamma])[0]
        return x_prop, y_prop

    def prop2x_syn(self, x_curr, y_curr):
        gamma = self.gamma
        xy_change = random.choices([-1,1],[gamma,1-gamma])[0]
        x_prop = x_curr + xy_change
        y_prop = y_curr + xy_change
        return x_prop, y_prop
