import numpy as np
from numpy.linalg import norm
from scipy import stats as stats


class Initialize(object):
    def __init__(self, mode, init_lag_steps=0, **kwargs):
        self.mode = mode
        self.init_lag_steps = init_lag_steps
        self.kwargs = kwargs

    def load_target(self, target): # for use when initializing from target
        self.target = target

    def load_prop(self, prop): # for use when the single kernel is used to make the init draw
        self.prop = prop

    def provide_inits(self):
        if not (hasattr(self, 'x_curr') and not hasattr(self, 'y_curr')):
            self.x_curr, self.y_curr = self._make_inits()

        return self.x_curr, self.y_curr

    def _make_inits(self):
        # this method can be overwritten in more specific classes,
        # as for the normal below
        mode = self.mode
        kwargs = self.kwargs

        if mode == 'fixed':
            x_curr = kwargs['x_curr']
            y_curr = kwargs['y_curr']

        elif mode == 'target_indep':
            assert self.target, 'target must be loaded to use this init mode'
            x_curr = self.target.rvs()
            y_curr = self.target.rvs()

        elif mode == 'target_ident':
            assert self.target, 'target must be loaded to use this init mode'
            x_curr = y_curr = self.target.rvs()

        else:
            raise ValueError('Initialization mode not supported')

        x_curr, y_curr = np.array(x_curr), np.array(y_curr)
        return x_curr, y_curr


class NormalInitialize(Initialize):
    @staticmethod
    def _curr_fn(d, r, m1, mr):
        """creates x,y from state variables"""
        x_curr = stats.norm.rvs(size=d)

        x_curr[0] = 0.
        x_curr = x_curr * mr / norm(x_curr)
        y_curr = x_curr.copy()

        x_curr[0] = m1 - r / 2.
        y_curr[0] = m1 + r / 2.
        return x_curr, y_curr

    def _make_inits(self):
        mode = self.mode
        kwargs = self.kwargs
        d = kwargs['d']

        if mode == 'fixed':
            x_curr = kwargs['x_curr']
            y_curr = kwargs['y_curr']

        elif mode == 'parametric':
            r = kwargs['r']
            m1 = kwargs['m1']
            mr = kwargs['mr']
            x_curr, y_curr = self._curr_fn(d, r, m1, mr)

        elif mode == 'target_indep':
            assert self.target, 'target must be loaded to use this init mode'
            x_curr = self.target.rvs()
            y_curr = self.target.rvs()

        elif mode == 'target_ident':
            assert self.target, 'target must be loaded to use this init mode'
            x_curr = y_curr = self.target.rvs()

        elif mode == 'offset_indep':
            init_norm = kwargs['init_norm']
            init_mu = init_norm * np.ones(d) / np.sqrt(d)
            x_curr, y_curr = stats.multivariate_normal.rvs(mean=init_mu, size=2)

        elif mode == 'offset_ident':
            d = kwargs['d']
            init_norm = kwargs['init_norm']
            init_mu = init_norm * np.ones(d) / np.sqrt(d)
            x_curr = y_curr = stats.multivariate_normal.rvs(mean=init_mu)

        else:
            raise ValueError('Initialization mode not supported')

        x_curr, y_curr = np.array(x_curr), np.array(y_curr)
        assert (len(x_curr) == d) & (len(y_curr) == d), 'invalid initial vector lengths'
        return x_curr, y_curr
