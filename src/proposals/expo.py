import numpy as np
from scipy import stats as stats

# need:
#   attributes: mode, d
#   methods: log_q_for_ar, log_prop1x_density, prop1x, prop2x

class ExpoProposal(object):
    def __init__(self, prop_mode, loc=1, scale=1, flip=True, **kwargs):
        self.d = 1
        self.flip = flip
        fl_coef = -1 if flip else 1
        self.fl_coef = fl_coef
        self.loc = loc
        self.scale = scale

        self.kwargs = kwargs
        self.mode = prop_mode

        # non-maximal couplings: independent, synchronous, and reflection
        if prop_mode == 'indep':
            self.prop2x = self.prop2x_indep
        elif prop_mode == 'syn':
            self.prop2x = self.prop2x_syn

        # simple maximal couplings: indep, reflected, parallel, and OT residuals
        elif prop_mode == 'max_indep':
            self.prop2x = self.prop2x_max_indep

        # full kernel couplings
        elif prop_mode == 'full_max_indep':
            self.prop2x = None

        else:
            raise ValueError('Proposal mode not supported')

    def log_q_for_ar(self, x_curr, x_next):
        diff = self.fl_coef*(x_next - x_curr)
        return stats.expon.pdf(x=diff, loc=-self.loc)

    def log_prop1x_density(self, x_curr, x_next):
        return self.log_q_for_ar(x_curr, x_next)

    def prop1x(self, x_curr):
        x_prop = x_curr + self.fl_coef * stats.expon.rvs(loc=self.fl_coef*self.loc,
                                                         scale=self.scale, size=1)
        return x_prop

    def prop2x_indep(self, x_curr):
        x_prop = x_curr + self.fl_coef * stats.expon.rvs(loc=self.fl_coef*self.loc,
                                                         scale=self.scale, size=1)
        y_prop = x_curr + self.fl_coef * stats.expon.rvs(loc=self.fl_coef*self.loc,
                                                         scale=self.scale, size=1)
        return x_prop, y_prop

    def prop2x_syn(self, x_curr, y_curr):
        """Synchronous coupling of Normals."""
        diff = self.fl_coef * stats.expon.rvs(loc=self.fl_coef*self.loc, size=1)
        x_prop = x_curr + diff
        y_prop = y_curr + diff
        return x_prop, y_prop

    # noinspection DuplicatedCode
    def prop2x_max_indep(self, x_curr, y_curr):
        x_prop = self.prop1x(x_curr)
        log_q_x_xp = self.log_prop1x_density(x_curr, x_prop)
        log_q_y_xp = self.log_prop1x_density(y_curr, x_prop)
        log_u = np.log(stats.uniform.rvs())
        if log_q_x_xp + log_u <= log_q_y_xp:
            return x_prop, x_prop
        else:
            accept = False
            while not accept:
                y_prop = self.prop1x(y_curr)
                log_q_x_yp = self.log_prop1x_density(x_curr, y_prop)
                log_q_y_yp = self.log_prop1x_density(y_curr, y_prop)
                log_v = np.log(stats.uniform.rvs())
                if log_q_y_yp + log_v > log_q_x_yp:
                    accept = True
            return x_prop, y_prop