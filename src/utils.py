
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler


# Pandas configuration

def configure_pd(pd):
    """Configure pandas for display"""
    pd.options.display.precision = 3
    pd.options.display.max_rows = 100
    pd.options.display.max_columns = 200
    pd.options.display.max_colwidth = 500
    pd.options.display.width = 100

# pyplot configuration

dark_blue = (0.04, 0.31, 0.59)
dark_red = (0.75, 0.10, 0.10)
dark_purple = (.4, .21, .35)  # midpoint of dark_blue and dark_red

col_list = [dark_blue, dark_red, 'forestgreen', 'goldenrod', 'purple', 'coral',
            'deeppink', 'lightskyblue', 'mistyrose', 'palegreen', 'navajowhite']


def configure_plt(plt, figsc=1.):
    """Set pyplot rcParams for attractive output"""
    plt_options = {'axes.grid': True,
                   'axes.axisbelow': True,
                   'figure.facecolor': 'white',
                   'axes.prop_cycle': cycler('color', col_list),
                   'figure.figsize': (11. * figsc, 8.5 * figsc),
                   'font.size': 18,
                   'grid.color': '.9',
                   'lines.color': dark_blue,
                   'lines.linewidth': 2.5,
                   'mathtext.default': 'regular'}

    plt.rcParams.update(plt_options)


# Other useful functions

def abline(intercept=0, slope=1):
    """Plot a line from slope and intercept, as in R"""
    axes = plt.gca()
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()

    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, linewidth=2)

    axes.set_xlim(xlim)
    axes.set_ylim(ylim)


# MCMC setup helper

def add_iter(par_list):
    for i, p in enumerate(par_list):
        p['i'] = i
        p['i_total'] = len(par_list)
    return par_list


# Covariance matrix generation - todo: migrate this to where its used

def make_weakcov(d, rho):
    cov_mtx = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            cov_mtx[i][j] = np.abs(i - j)
    cov_mtx = np.power(rho, cov_mtx)
    cov_mtx_inv = np.linalg.inv(cov_mtx)
    return cov_mtx, cov_mtx_inv
