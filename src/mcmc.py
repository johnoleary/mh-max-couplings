
import hashlib
import time

import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy.linalg import norm


class Recorder(object):
    # record initialization, updating, and output methods

    def _init_record(self):
        #record_items = self.record_items

        # we're going to initialize everything, but not record everything on the fly
        # the np.nans below are needed for iter count to line up.

        # full state of x and y
        self.x_chain, self.y_chain = [self.x_curr], [self.y_curr]

        # Ax and Ay
        self.x_chain_acc, self.y_chain_acc = [np.nan], [np.nan]

        # Acceptance probabilities
        self.prob_x_list, self.prob_y_list, self.prop_meeting = [np.nan], [np.nan], [np.nan]

        # r = ||y-x||
        self.r_curr_list, self.r_next_list = [np.nan], [np.nan]

        # nmet = #{i : x_i = y_i}
        self.nmet_curr_list, self.nmet_next_list = [np.nan], [np.nan]

        # ||m|| for m = (y+x)/2
        self.m_curr_list, self.m_next_list = [np.nan], [np.nan]

        # m1 = e'm
        self.m1_curr_list, self.m1_next_list = [np.nan], [np.nan]

        # x_prop = x_curr + xi1
        self.xi1_list = [np.nan]

        # under optimal transport accept/reject, which was chosen?
        self.ar_choice_list = [np.nan]

        # try counts when there is a required pattern
        self.try_count_list = [np.nan]

        # what does this do?
        self.eps_r_list, self.eps_r_half_list = [np.nan], [np.nan]

        # acceptance scenario probabilities
        # self.prob_one_list, self.prob_both_list = [np.nan], [np.nan]

        # area of the parallelogram with sides x,y
        self.area_curr_list, self.area_next_list = [np.nan], [np.nan]


    # Record results at each iteration
    def _record(self, which_var=('x', 'y')):
        assert set(which_var).issubset({'x', 'y'})
        record_items = self.record_items

        x_curr, y_curr = self.x_curr, self.y_curr
        x_prop, y_prop = self.x_prop, self.y_prop
        x_next, y_next = self.x_next, self.y_next
        xi_prop = x_prop - x_curr

        m_curr, m_next = (x_curr + y_curr) / 2., (x_next + y_next) / 2.
        r_curr, r_next = norm(y_curr - x_curr), norm(y_next - x_next)
        nmet_curr, nmet_next = sum(np.isclose(y_curr, x_curr)), sum(np.isclose(y_next, x_next))

        if r_curr < 1e-8:
            e_curr = np.zeros(self.d)
        else:
            e_curr = (y_curr - x_curr) / r_curr

        if r_next < 1e-8:
            e_next = np.zeros(self.d)
        else:
            e_next = (y_next - x_next) / r_next

        m1_curr = e_curr.dot(m_curr)
        m1_next = e_next.dot(m_next)

        # record chain position
        if 'chain_positions' in record_items:
            if 'x' in which_var:
                self.x_chain.append(x_curr)
            if 'y' in which_var:
                self.y_chain.append(y_curr)

        # record accept/reject
        if 'accrej' in record_items:
            if 'x' in which_var:
                self.x_chain_acc.append(self.x_acc)
            if 'y' in which_var:
                self.y_chain_acc.append(self.y_acc)

        # record accept/reject mode (i.e. for opt transport method)
        if 'ar_choice' in record_items:
            self.ar_choice_list.append(self.ar_choice)

        if 'accrej_prob' in record_items:
            #larx = self._log_acc_ratio(x_curr, x_prop)
            #lary = self._log_acc_ratio(y_curr, y_prop)

            #pax = min(1., np.exp(larx))
            #pay = min(1., np.exp(lary))
            #pax = np.exp(larx)
            #pay = np.exp(lary)

            #p_one = np.abs(pax - pay)
            #p_both = min(pax, pay)

            #self.prob_one_list.append(p_one)
            #self.prob_both_list.append(p_both)
            self.prob_x_list.append(self.x_acc_prob)
            self.prob_y_list.append(self.y_acc_prob)
            self.prop_meeting.append(np.allclose(x_prop,y_prop))

        # record distance between chains
        if 'r' in record_items:
            self.r_curr_list.append(r_curr)
            self.r_next_list.append(r_next)

        # record nmet
        if 'nmet' in record_items:
            self.nmet_curr_list.append(nmet_curr)
            self.nmet_next_list.append(nmet_next)

        # record m_1
        if 'm1' in record_items:
            self.m1_curr_list.append(m1_curr)
            self.m1_next_list.append(m1_next)

        if 'area' in record_items:
            V_curr = np.stack([x_curr, y_curr])
            V_next = np.stack([x_next, y_next])
            area_curr = np.linalg.det(np.inner(V_curr, V_curr))
            area_next = np.linalg.det(np.inner(V_next, V_next))

            self.area_curr_list.append(area_curr)
            self.area_next_list.append(area_next)

        # record ||m_{-1}||
        if 'm' in record_items:
            self.m_curr_list.append(norm(m_curr))
            self.m_next_list.append(norm(m_next))

        # record xi1
        if 'xi1' in record_items:
            xi1_prop = e_curr.dot(xi_prop)
            self.xi1_list.append(xi1_prop)

        # record try count for specific acceptance patterns
        if 'tries' in record_items:
            self.try_count_list.append(self.try_count)

        # record ||m_{-1} + xi_{-1}||^2 - ||m_{-1}||^2
        if 'eps_r' in record_items:
            mr_curr = m_curr - e_curr * e_curr.dot(m_curr)
            xi_r_prop = xi_prop - e_curr * e_curr.dot(xi_prop)

            eps_r_prop = norm(mr_curr + xi_r_prop) ** 2 - norm(mr_curr) ** 2
            self.eps_r_list.append(eps_r_prop)

            eps_r_prop_half = norm(mr_curr + xi_r_prop / 2.) ** 2 - norm(mr_curr) ** 2
            self.eps_r_half_list.append(eps_r_prop_half)

    def _form_result_dict(self, trim_iter_0=True):
        record_items = self.record_items

        out = {'t': self.x_iter}  # 'iter' is a df method so we use t

        if 'chain_positions' in record_items:
            out.update({'x_chain':self.x_chain, 'y_chain':self.y_chain})
        if 'accrej' in record_items:
            out.update({'Ax': self.x_chain_acc, 'Ay': self.y_chain_acc})
        if 'accrej_prob' in record_items:
            #out.update({'p_both': self.prob_both_list, 'p_one': self.prob_one_list})
            out.update({'x_acc_prob': self.prob_x_list, 'y_acc_prob': self.prob_y_list,
                        'prop_meeting': self.prop_meeting})
        if 'ar_choice' in record_items:
            out.update({'ar_choice': self.ar_choice_list})
        if 'tries' in record_items:
            out.update({'tries': self.try_count_list})
        if 'r' in record_items:
            out.update({'r_curr': self.r_curr_list, 'r_next': self.r_next_list})
        if 'nmet' in record_items:
            out.update({'nmet_curr': self.nmet_curr_list, 'nmet_next': self.nmet_next_list})
        if 'm1' in record_items:
            out.update({'m1_curr': self.m1_curr_list, 'm1_next': self.m1_next_list})
        if 'm' in record_items:
            out.update({'m_curr': self.m_curr_list, 'm_next': self.m_next_list})
        if 'area' in record_items:
            out.update({'area_curr': self.area_curr_list,
                        'area_next': self.area_next_list})
        if 'xi1' in record_items:
            out.update({'xi1_prop': self.xi1_list})
        if 'eps_r' in record_items:
            out.update({'eps_r_prop': self.eps_r_list, 'eps_r_prop_half': self.eps_r_half_list})

        if trim_iter_0:
            out = {k: v[1:] for k, v in out.items()}

        out.update({'d': self.d})

        self.result_dict = out

    def return_df(self):
        df = pd.concat(pd.DataFrame(res) for res in self.rep_result_list)

        if 'accrej' in self.record_items:
            df['cx'] = df['Ax']
            df['cy'] = df['Ay']
            df['cb'] = df['Ax'] & df['Ay']
            df['cs'] = df['Ax'] ^ df['Ay']
            df.drop(columns=['Ax','Ay'], inplace=True)

        if 'tries' in self.record_items:
            df['maxed_tries'] = (df.tries == self.max_tries)

        if 'm1' in self.record_items:
            df['abs_m1_curr'] = np.abs(df['m1_curr'])
            df['abs_m1_next'] = np.abs(df['m1_next'])

        if 'chain_positions' in self.record_items:
            df_x_chain = pd.DataFrame(df['x_chain'].tolist(), index=df.index)
            df_y_chain = pd.DataFrame(df['y_chain'].tolist(), index=df.index)
            df_x_chain.columns = [f'x{c}' for c in df_x_chain.columns]
            df_y_chain.columns = [f'y{c}' for c in df_y_chain.columns]
            df = df.drop(columns=['x_chain', 'y_chain'])
            df = pd.concat([df, df_x_chain, df_y_chain], axis=1)

        return df


class FullKernel(object):
    # full kernel functions

    def _transport_fn(self, x_curr, y_curr, x_next, trans_mode):
        if trans_mode=='refl':
            r_curr = norm(y_curr - x_curr)
            if r_curr < 1e-8:
                e_curr = np.zeros(self.d)
            else:
                e_curr = (y_curr - x_curr) / r_curr
            xi_prop = x_next - x_curr
            eta_prop = xi_prop - 2 * e_curr * e_curr.dot(xi_prop)
            y_next = y_curr + eta_prop  # reflection of x move
        elif trans_mode=='sync':
            xi_prop = x_next - x_curr
            eta_prop = xi_prop
            y_next = y_curr + eta_prop  # reflection of x move
        else:
            raise ValueError("Invalid transport mode")
        return y_next

    def _r_xy_fn(self, x_curr, y_curr, x_next):
        # kernel residual \tilde f^r_{xy}(x')
        p_x_zp = self._kernel_density(x_curr, x_next)
        p_y_zp = self._kernel_density(y_curr, x_next)
        return max(0, p_x_zp - p_y_zp)

    def _tr_xy_fn(self, x_curr, y_curr, x_next, trans_mode):
        # kernel reflection residual \tilde f^t_{xy}(x')
        y_refl = self._transport_fn(x_curr, y_curr, x_next, trans_mode)
        r_xy_xp = self._r_xy_fn(x_curr, y_curr, x_next)
        r_yx_yp = self._r_xy_fn(y_curr, x_curr, y_refl)
        return max(0, r_xy_xp - r_yx_yp)

    def _draw_kernel_res(self, x_curr, y_curr):
        """Algorithm 3 in Guanyang Wang's note, for full max indep coupling"""
        accept = False
        while not accept:
            y_prop, y_acc, y_next = self._iter1x_step(y_curr)
            if not y_acc:
                return y_curr
            log_v = np.log(stats.uniform.rvs())
            log_p_y_yp = self._log_kernel_density(y_curr, y_prop)
            log_p_x_yp = self._log_kernel_density(x_curr, y_prop)
            if log_v + log_p_y_yp >= log_p_x_yp:
                return y_prop

    def _draw_kernel_trans_res(self, x_curr, y_curr, trans_mode='refl'):
        """Algorithm 5 in Guanyang's note"""
        accept = False
        while not accept:
            y_prop, y_acc, y_next = self._iter1x_step(y_curr)
            if not y_acc:
                return y_curr

            v = stats.uniform.rvs()
            p_y_yp = self._kernel_density(y_curr, y_next)
            pres_y_yp = self._tr_xy_fn(y_curr, x_curr, y_next,
                                       trans_mode=trans_mode)
            if v <= min(1, pres_y_yp/p_y_yp):
                return y_next

    def _iter2x_full_max_indep(self, x_curr, y_curr):
        """Algorithm 4 in Guanyang Wang's note"""
        x_prop = self.prop1x(x_curr)
        log_u = np.log(stats.uniform.rvs())
        x_acc = (log_u <= self._log_acc_ratio(x_curr, x_prop))
        x_next = x_acc * x_prop + (1 - x_acc) * x_curr

        if x_acc:
            log_v = np.log(stats.uniform.rvs())
            log_p_x_xp = self._log_kernel_density(x_curr, x_next)
            log_p_y_xp = self._log_kernel_density(y_curr, x_next)

            log_p_ratio = min(0, log_p_y_xp - log_p_x_xp)
            if log_v <= log_p_ratio:
                y_next = x_next
            else:
                y_next = self._draw_kernel_res(x_curr, y_curr)
        else:
            y_next = self._draw_kernel_res(x_curr, y_curr)

        return x_next, y_next, True, True, x_next, y_next

    def _iter2x_full_max_trans(self, x_curr, y_curr, trans_mode='refl'):
        """Algorithm 6 in Guanyang Wang's note"""

        # Step 1. sample x' ~ P(x,.)
        x_prop, x_acc, x_next = self._iter1x_step(x_curr)
        if not x_acc:
            y_next = self._draw_kernel_trans_res(x_curr, y_curr, trans_mode=trans_mode)
        else:
            log_v = np.log(stats.uniform.rvs())
            log_p_y_xp = self._log_kernel_density(y_curr, x_next)
            log_p_x_xp = self._log_kernel_density(x_curr, x_next)
            if x_acc and (log_v + log_p_x_xp <= log_p_y_xp):
                y_next = x_next
            else:
                y_refl = self._transport_fn(x_curr, y_curr, x_next, trans_mode=trans_mode)
                w = stats.uniform.rvs()
                pres_x_xp = self._tr_xy_fn(x_curr, y_curr, x_next, trans_mode=trans_mode)
                pres_y_xp = self._tr_xy_fn(y_curr, x_curr, y_refl, trans_mode=trans_mode)
                if x_acc and w*pres_x_xp <= pres_y_xp:
                    y_next = y_refl

                else:
                    y_next = self._draw_kernel_trans_res(x_curr, y_curr, trans_mode=trans_mode)

        return x_next, y_next, True, True, x_next, y_next


class CoupledMH(FullKernel, Recorder):
    def __init__(self, init, prop, target,
                 ar_mode='same_u',
                 record_items=('r','accrej'),
                 required_pattern='any',
                 max_tries=np.inf,
                 break_condition='none'):
        """
        required_pattern in ('any', 'cb','cs','cx','cy', 'not_cs')
        max_tries: number of tries allowed to get the pattern.
        """

        np.random.seed() # needed for multiprocessing

        # target distribution

        self.d = target.d
        self.log_pi_for_ar = target.log_pi_for_ar

        # proposal distribution
        self.prop1x = prop.prop1x
        self.prop2x = prop.prop2x
        self.log_q_for_ar = prop.log_q_for_ar
        self.prop_log_density = prop.log_prop1x_density
        self.prop_mode = prop.mode
        assert prop.d == target.d, "target and proposal dimension mismatch"

        # initial distribution
        init.load_target(target)
        self.init = init
        self.init_lag_steps = init.init_lag_steps

        # mcmc control
        self.ar_mode = ar_mode
        self.required_pattern = required_pattern
        self.max_tries = max_tries
        self._process_bc(break_condition)

        # record items
        if type(record_items) is str:
            record_items = [record_items]

        self.record_items = set(record_items)

    #define break condition dict, if necessary
    def _process_bc(self, break_condition):
        if type(break_condition) is str:
            self.break_condition = {'name': break_condition}
        elif type(break_condition) is dict:
            self.break_condition = break_condition
        else:
            raise TypeError('break_condition must be str or dict')

    # Acceptance ratios and transition kernel densities
    def _log_acc_ratio(self, z_curr, z_prop):
        log_pi_for_ar = self.log_pi_for_ar
        log_q_for_ar = self.log_q_for_ar
        log_pi_x, log_pi_xp = log_pi_for_ar(z_curr), log_pi_for_ar(z_prop)
        log_q_x_xp, log_q_xp_x = log_q_for_ar(z_curr, z_prop), log_q_for_ar(z_prop, z_curr)
        log_a = min(0, log_pi_xp - log_pi_x + log_q_xp_x - log_q_x_xp)
        return log_a

    def _log_kernel_density(self, z_curr, z_prop):
        """the log density p(z,z') for the continuous part
        of the full MH transition kernel. """

        assert not np.allclose(z_curr, z_prop, rtol=1e-08)
        log_prop_density = self.prop_log_density(z_curr, z_prop)
        log_acc_ratio = self._log_acc_ratio(z_curr, z_prop)
        return log_prop_density + log_acc_ratio

    def _kernel_density(self, z_curr, z_prop):
        return np.exp(self._log_kernel_density(z_curr, z_prop))

    # Single iteration
    def _iter1x_step(self, z_curr):
        z_prop = self.prop1x(z_curr)
        log_u = float(np.log(stats.uniform.rvs()))
        log_acc_ratio_pt = float(self._log_acc_ratio(z_curr, z_prop))
        z_acc = (log_u <= log_acc_ratio_pt)
        z_next = z_acc * z_prop + (1 - z_acc) * z_curr
        return z_prop, z_acc, z_next

    def iterate_1x(self, n, which_var):
        """
        Update a single chain, as specified by which_var.
        """
        assert which_var in ('x', 'y'), 'variable to iterate must be x or y'
        z_curr = self.x_curr if which_var == 'x' else self.y_curr
        for i in range(n):
            z_prop, z_acc, z_next = self._iter1x_step(z_curr)
            z_curr = z_next

        if which_var == 'x':
            self.x_curr = z_curr
        else:
            self.y_curr = z_curr

    def _acceptreject_2x(self, x_curr, y_curr, x_prop, y_prop):
        """
        Produce x and y acceptance indicators based on the current and proposed
        states, according to a specified accept/reject mode.
        """
        ar_mode = self.ar_mode

        if ar_mode in ('same_u', 'refl_u', 'indep_u', 'ot_u'):
            # draw u,v according to the specified mode
            if ar_mode == 'same_u':
                u = stats.uniform.rvs()
                v = u
                self.ar_choice = 'same_u'

            elif ar_mode == 'refl_u':
                u = stats.uniform.rvs()
                v = 1. - u
                self.ar_choice = 'refl_u'

            elif ar_mode == 'indep_u':
                u, v = stats.uniform.rvs(size=2)
                self.ar_choice = 'indep_u'

            elif ar_mode == 'ot_u':
                dn = norm(y_curr - x_curr)
                dx = norm(y_curr - x_prop)
                dy = norm(y_prop - x_curr)
                db = norm(y_prop - x_prop)
                u = stats.uniform.rvs()
                if dn + db <= dx + dy:
                    v = u
                    self.ar_choice = 'same_u'
                else:
                    v = 1. - u
                    self.ar_choice = 'refl_u'

            # translate to accept/reject decision
            log_u, log_v = float(np.log(u)), float(np.log(v))
            log_acc_ratio_x = float(self._log_acc_ratio(x_curr, x_prop))
            log_acc_ratio_y = float(self._log_acc_ratio(y_curr, y_prop))

            x_acc = (log_u <= log_acc_ratio_x)
            y_acc = (log_v <= log_acc_ratio_y)

            self.x_acc_prob = log_acc_ratio_x
            self.y_acc_prob = log_acc_ratio_y

        elif ar_mode in ('cond_same_u','cond_refl_u','cond_indep_u'):
            eps = np.finfo(float).eps

            def log_q(z_curr, z_prop):
                return self.prop_log_density(z_curr, z_prop)

            def log_p(z_curr, z_prop):
                return self._log_kernel_density(z_curr, z_prop)

            def q(z_curr, z_prop):
                return np.exp(log_q(z_curr, z_prop))

            def p(z_curr, z_prop):
                return np.exp(log_p(z_curr, z_prop))

            def log_qm(x_curr, y_curr, x_prop):
                return min(log_q(x_curr, x_prop), log_q(y_curr, x_prop))

            def qm(x_curr, y_curr, x_prop):
                return np.exp(log_qm(x_curr, y_curr, x_prop))

            def qr(x_curr, y_curr, x_prop):
                return q(x_curr, x_prop) - qm(x_curr, y_curr, x_prop)

            def log_qr(x_curr, y_curr, x_prop):
                return np.log(max(qr(x_curr, y_curr, x_prop),eps))

            def pr(x_curr, y_curr, x_prop):
                return max(p(x_curr, x_prop) - qm(x_curr, y_curr, x_prop), 0)

            def log_pr(x_curr, y_curr, x_prop):
                return np.log(max(pr(x_curr, y_curr, x_prop), eps)) #o/w requires elaborate cases

            u = stats.uniform.rvs()
            log_u = np.log(u)

            # compute log acceptance ratios
            lar_x_meet = float(log_p(x_curr, x_prop) - log_qm(x_curr, y_curr, x_prop))
            lar_y_meet = float(log_p(y_curr, y_prop) - log_qm(y_curr, x_curr, y_prop))
            lar_x_not = float(log_pr(x_curr, y_curr, x_prop) - log_qr(x_curr, y_curr, x_prop))
            lar_y_not = float(log_pr(y_curr, x_curr, y_prop) - log_qr(y_curr, x_curr, y_prop))

            if np.allclose(x_prop, y_prop):
                x_acc = (float(log_u) <= lar_x_meet)
                y_acc = (float(log_u) <= lar_y_meet)

                self.x_acc_prob = lar_x_meet
                self.y_acc_prob = lar_y_meet

            else:
                if ar_mode=='cond_same_u':
                    v = u
                elif ar_mode=='cond_refl_u':
                    v = 1-u
                elif ar_mode=='cond_indep_u':
                    v = stats.uniform.rvs()

                log_v = np.log(v)
                x_acc = (float(log_u) <= lar_x_not)
                y_acc = (float(log_v) <= lar_y_not)

                self.x_acc_prob = lar_x_not
                self.y_acc_prob = lar_y_not

        else:
            raise ValueError('Invalid ar_mode')
            # the one-step full kernel couplings won't have one of
            # these modes, but they shouldn't ever call this method.

        return x_acc, y_acc

    # Coupled iteration
    def _iter2x_step(self, x_curr, y_curr):
        """one step under the usual method of drawing from a coupled proposal and
        then a coupled accept/reject step
        """
        if np.allclose(x_curr, y_curr):
            x_prop, x_acc, x_next = y_prop, y_acc, y_next = self._iter1x_step(x_curr)
        else:
            x_prop, y_prop = self.prop2x(x_curr, y_curr)
            x_acc, y_acc = self._acceptreject_2x(x_curr, y_curr, x_prop, y_prop)

            x_next = x_acc * x_prop + (1 - x_acc) * x_curr
            y_next = y_acc * y_prop + (1 - y_acc) * y_curr

        return x_prop, y_prop, x_acc, y_acc, x_next, y_next

    def iterate_2x(self, n_iter=1):
        """Update the x and y chains."""

        for i in range(n_iter):
            x_it, y_it = self.x_iter[-1], self.y_iter[-1]
            # assert x_it == y_it + 1, "Incompatible x and y iter count"
            assert x_it == y_it, "Incompatible x and y iter count"

            if self.prop_mode not in ('full_max_indep', 'full_max_refl', 'full_max_sync'):
                # i.e. if we are in one of the usual two stage couplings
                repeat,try_count = True, 0 #for the 'required pattern' options
                while repeat and (try_count < self.max_tries):
                    try_count += 1
                    res = self._iter2x_step(self.x_curr, self.y_curr)
                    x_prop, y_prop, x_acc, y_acc, x_next, y_next = res
                    repeat = self._pattern_try_again(x_acc, y_acc)
                self.try_count = try_count

            else: #full kernel couplings
                if self.prop_mode == 'full_max_indep':
                    res = self._iter2x_full_max_indep(self.x_curr, self.y_curr)
                elif self.prop_mode == 'full_max_refl':
                    res = self._iter2x_full_max_trans(self.x_curr, self.y_curr, trans_mode='refl')
                elif self.prop_mode == 'full_max_sync':
                    res = self._iter2x_full_max_trans(self.x_curr, self.y_curr, trans_mode='sync')
                x_prop, y_prop, x_acc, y_acc, x_next, y_next = res

            self.x_prop, self.y_prop = x_prop, y_prop
            self.x_acc, self.y_acc = x_acc, y_acc
            self.x_next, self.y_next = x_next, y_next
            self.x_iter.append(x_it + 1)
            self.y_iter.append(y_it + 1)
            self._record()

            self.x_curr, self.y_curr = x_next, y_next

            if self._test_break_condition():
                break

        if (i == n_iter-1) and self.break_condition['name']=='meet':
            print("Warning: chains did not meet by max_iter")

    def _test_break_condition(self):
        break_condition = self.break_condition
        bcn = break_condition['name']

        if bcn == 'none':
            return False
        elif bcn == 'meet':
            return np.allclose(self.x_curr, self.y_curr)
        elif bcn == 'after_first_cs':
            return self.x_acc ^ self.y_acc
        elif bcn == 'abs_m1_threshold':
            return np.abs(self.m1_next_list[-1] < break_condition['threshold'])
        elif bcn == 'r_threshold':
            return np.abs(self.r_next_list[-1] < break_condition['threshold'])
        else:
            raise ValueError("Invalid break condition")

    def _pattern_try_again(self, x_acc, y_acc):
        required_pattern = self.required_pattern

        if required_pattern == 'any':
            return False
        elif (required_pattern == 'cb') & (x_acc & y_acc):
            return False
        elif (required_pattern == 'cs') & (x_acc ^ y_acc):
            return False
        elif (required_pattern == 'cx') & (x_acc & ~y_acc):
            return False
        elif (required_pattern == 'cy') & (~x_acc & y_acc):
            return False
        elif (required_pattern == 'not_cs') & ~(x_acc ^ y_acc):
            return False
        else:
            return True

    def _one_run(self, n_iter):
        self._init_state()
        self._init_record()

        if self.init_lag_steps>0:
            self.iterate_1x(n=self.init_lag_steps, which_var='y')

        self.iterate_2x(n_iter=n_iter)
        self._form_result_dict()

    def _init_state(self):
        self.x_iter, self.y_iter = [0], [0]
        self.x_curr, self.y_curr = self.init.provide_inits()
        self.x_next, self.y_next = np.nan, np.nan
        self.x_prop, self.y_prop = np.nan, np.nan
        self.x_acc, self.y_acc = True, True
        self.x_acc_prob, self.y_acc_prob = np.nan, np.nan
        #self.ar_choice = ''

    @staticmethod
    def _print_iter(i, ntotal, pr_postfix):
        incr = int(10**max(0, np.floor(np.log10(ntotal)) - 1))
        if (i == ntotal) or (i % incr == 0):
            print(f'{i}{pr_postfix}') #, end=' ')

    def run(self, n_iter, n_rep=1, verbose=True, pr_postfix=""):
        rep_result_list = []

        for i in range(n_rep):
            if verbose:
                self._print_iter(i+1, n_rep, pr_postfix)
            self._one_run(n_iter)

            # determine rep_id
            hash = hashlib.sha1()
            hash.update(str(time.time()).encode('utf-8'))
            self.result_dict.update({'rep_id': hash.hexdigest()[:10]})

            rep_result_list.append(self.result_dict)

        self.rep_result_list = rep_result_list

        #print('\n' if verbose else '')

