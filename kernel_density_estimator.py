# kernel_density_estimator.py

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import norm, expon

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

plt.rcParams["figure.figsize"] = (15, 10)


# ===================================================
# kde_pdf and kde_cdf are used for compiling kernel |
# density and distribution estimates.               |
# ===================================================

def kde_pdf(data, kernel_func, bandwidth):
    """Generate kernel density estimator over data."""
    kernels = dict()
    n = len(data)
    for d in data:
        kernels[d] = kernel_func(d, bandwidth)

    def evaluate(x):
        """Evaluate `x` using kernels above."""
        pdfs = list()
        for d in data:
            pdfs.append(kernels[d](x))
        return sum(pdfs) / n

    return evaluate


# ============================================
# Uniform Kernel PDF                         |
# ============================================

def uniform_pdf(x_i, bandwidth):
    """Return uniform kernel density estimator."""
    lowerb = (x_i - bandwidth)
    upperb = (x_i + bandwidth)

    def evaluate(x):
        """Evaluate x."""
        if x <= lowerb:
            pdf = 0
        elif x > upperb:
            pdf = 0
        else:
            pdf = (1 / (2 * bandwidth))
        return pdf

    return evaluate


# ============================================
# Epanechnikov Kernel PDF                      |
# ============================================

def epanechnikov_pdf(x_i, bandwidth):
    """Return epanechnikov kernel density estimator."""
    lowerb = (x_i - bandwidth)
    upperb = (x_i + bandwidth)

    def evaluate(x):
        """Evaluate x."""
        if x <= lowerb:
            pdf = 0
        elif x > upperb:
            pdf = 0
        else:
            pdf = ((3 * (bandwidth ** 2 - (x - x_i) ** 2)) / (4 * bandwidth ** 3))
        return pdf

    return evaluate


# ============================================
# Gaussian Kernel PDF                        |
# ============================================
def gaussian_pdf(x_i, bandwidth):
    """Return Gaussian kernel density estimator."""
    x_bar = x_i

    def evaluate(x):
        """Evaluate x."""
        pdf = (np.sqrt(2 * np.pi * bandwidth ** 2) ** -1) * np.exp(-((x - x_bar) ** 2) / (2 * bandwidth ** 2))
        return pdf

    return evaluate


# =========================================
# kernel density estimates visualizations |
# =========================================
def plot_kde(true_dist, num_samples, kernel_function, bandwidth_h):
    vals = np.array([])

    if true_dist == 'exp':
        vals = np.random.exponential(1, num_samples)
    elif true_dist == 'norm':
        vals = np.random.normal(0, 1, num_samples)

    xvals = np.arange(min(vals), max(vals), .01)

    # ========================================================
    # Bandwidth Selection : rule-of-thumb plugin             |
    # ========================================================
    if "uniform_pdf" in str(kernel_function):
        sigma_hat = np.std(vals)
        R_k = 1 / 2
        kappa_2 = 1 / 3
        h_opt = (((8 * (np.pi ** 0.5) * R_k) / (3 * kappa_2 * num_samples)) ** 0.2) * sigma_hat

    elif "epanechnikov_pdf" in str(kernel_function):
        sigma_hat = np.std(vals)
        R_k = 3 / 5
        kappa_2 = 1 / 5
        h_opt = (((8 * (np.pi ** 0.5) * R_k) / (3 * kappa_2 * num_samples)) ** 0.2) * sigma_hat

    elif "gaussian_pdf" in str(kernel_function):
        sigma_hat = np.std(vals)
        R_k = 1 / (2 * (np.pi ** 0.5))
        kappa_2 = 1
        h_opt = (((8 * (np.pi ** 0.5) * R_k) / (3 * kappa_2 * num_samples)) ** 0.2) * sigma_hat

    # ========================================================
    # Bandwidth Selection : cross-validation                 |
    # ========================================================
    grid = GridSearchCV(KernelDensity(), {'bandwidth': xvals}, cv=20)
    grid.fit(vals[:, None])
    h_cv = grid.best_params_["bandwidth"]

    # ========================================================
    # Optimized Bandwidth visualization                      |
    # ========================================================
    fig = plt.figure()
    ax4 = fig.add_subplot(2, 2, 1)
    dist_4 = kde_pdf(vals, kernel_func=kernel_function, bandwidth=h_opt)
    y4 = [dist_4(i) for i in xvals]
    ys4 = [dist_4(i) for i in vals]
    ax4.scatter(vals, ys4)
    if true_dist == 'exp':
        ax4.plot(xvals, expon.pdf(xvals))
    elif true_dist == 'norm':
        ax4.plot(xvals, norm.pdf(xvals, 0, 1))
    ax4.plot(xvals, y4)

    # bandwidth=optimal_bandwidth_crossvalidated:
    ax5 = fig.add_subplot(2, 2, 2)
    dist_5 = kde_pdf(vals, kernel_func=kernel_function, bandwidth=h_cv)
    y5 = [dist_5(i) for i in xvals]
    ys5 = [dist_5(i) for i in vals]
    ax5.scatter(vals, ys5)
    if true_dist == 'exp':
        ax5.plot(xvals, expon.pdf(xvals))
    elif true_dist == 'norm':
        ax5.plot(xvals, norm.pdf(xvals, 0, 1))
    ax5.plot(xvals, y5)

    # display gridlines
    g4 = ax4.grid(True)
    g5 = ax5.grid(True)

    # display legend in each subplot
    leg4 = mpatches.Patch(color=None, label=f'plug-in bandwidth={h_opt}')
    leg5 = mpatches.Patch(color=None, label=f'cross-validated bandwidth={h_cv}')

    ax4.legend(handles=[leg4])
    ax5.legend(handles=[leg5])

    plt.tight_layout()
    plt.show()
