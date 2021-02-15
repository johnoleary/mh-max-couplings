
class Proposal(object):
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
