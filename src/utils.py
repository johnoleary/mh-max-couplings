import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler


# Pandas configuration

def configure_pd(pd):
    """
    Configures pandas for nice display in Jupyter notebooks.

    Args:
        pd: an instance of the pandas module.

    Returns:
        None

    """
    pd.options.display.precision = 3
    pd.options.display.max_rows = 100
    pd.options.display.max_columns = 200
    pd.options.display.max_colwidth = 500
    pd.options.display.width = 100


# pyplot configuration

dark_blue = (0.04, 0.31, 0.59)
dark_red = (0.75, 0.10, 0.10)
dark_purple = (0.4, 0.21, 0.35)  # midpoint of dark_blue and dark_red

col_list = [dark_blue, dark_red, 'forestgreen', 'goldenrod', 'purple', 'coral',
            'deeppink', 'lightskyblue', 'mistyrose', 'palegreen', 'navajowhite']


def configure_plt(plt_input, figsc: float = 1.):
    """
    Sets pyplot rcParams for attractive output.

    Args:
        plt_input: an instance of the matplotlib.pyplot module
        figsc (optional): a scale factor for making output images larger or smaller

    Returns:
        None
    """

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

    plt_input.rcParams.update(plt_options)


# Other useful functions

def abline(intercept: float = 0., slope: float = 1) -> None:
    """
    Plots a line from slope and intercept, as in R.

    Args:
        intercept: a float giving the y intercept of the line to be plotted
        slope: a float giving the slope of the line to be plotted

    Returns:
        None
    """

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
    """
    Adds iteration number and total number of iterations to parameter list for use by module mcmc.

    Args:
        par_list: parameter list object

    Returns:
        list: parameter list with iteration # and total iterations added
    """
    for i, p in enumerate(par_list):
        p['i'] = i
        p['i_total'] = len(par_list)
    return par_list


# Covariance matrix generation

def make_weakcov(d: int, rho: float):
    """
    Creates covariance matrix with ij entry rho^|i-j|.

    Args:
        d: dimension of the space.
        rho: correlation parameter.

    Returns:
        tuple of ndarray: d x d covariance matrix and its inverse.
    """
    cov_mtx = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            cov_mtx[i][j] = np.abs(i - j)
    cov_mtx = np.power(rho, cov_mtx)
    cov_mtx_inv = np.linalg.inv(cov_mtx)
    return cov_mtx, cov_mtx_inv
