# kernel_density_estimator.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import norm, expon
from random import randrange


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


def cross_validation_split(dataset, folds=10):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def bundle_test_train_set(dataset, k, test_idx):
    folds = cross_validation_split(dataset, folds=k)
    test = np.array(folds[test_idx])

    train = list()
    for i, x in enumerate(folds):
        if i != test_idx:
            train.extend(folds[i])
    train = np.array(train)
    return test, train


def estimate_bandwidth(data, kernel_function, true_dist):
    # ========================================================
    # Bandwidth Selection : cross validation method          |
    # ========================================================
    bandwidths = np.arange(0.01, 2, 0.02)
    if true_dist == 'exp':
        pdf = expon.pdf(data)
    elif true_dist == 'norm':
        pdf = norm.pdf(data)

    data = np.array(zip(pdf, data))
    # estimate y_hat corresponding to X
    errors = list()
    k = 10
    for h in bandwidths:
        error = 0
        for i in range(k):
            test, train = bundle_test_train_set(data, k, i)
            estimator = kde_pdf(train, kernel_func=kernel_function, bandwidth=h)
            y_hat = [estimator(x) for x in test[:, 1]]
            error += (test[:, 0] - y_hat) ** 2
        errors.append(sum(error))

    errors = np.array(errors)
    h_opt = bandwidths[np.argmin(errors)]
    return h_opt


def calculate_optimum_bandwidth(vals, kernel_function):
    # ========================================================
    # Bandwidth Selection : rule-of-thumb plugin             |
    # ========================================================
    num_samples = len(vals)
    h_opt = 0
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

    return h_opt


# =========================================
# kernel density estimates visualizations |
# =========================================
def plot_kde(true_dist, num_samples, kernel_function):
    vals = np.array([])
    if true_dist == 'exp':
        vals = np.random.exponential(1, num_samples)
    elif true_dist == 'norm':
        vals = np.random.normal(0, 1, num_samples)

    xvals = np.arange(min(vals), max(vals), .01)

    h_opt = calculate_optimum_bandwidth(vals, kernel_function)

    # ========================================================
    # Bandwidth Selection : cross-validation                 |
    # ========================================================
    h_cv = estimate_bandwidth(xvals, gaussian_pdf, true_dist)

    # ========================================================
    # Optimized Bandwidth visualization                      |
    # ========================================================
    fig = plt.figure()
    # plugin optimal bandwidth
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

    # bandwidth chosen from cross validation
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
    ax4.grid(True)
    ax5.grid(True)

    # display legend in each subplot
    leg4 = mpatches.Patch(color=None, label=f'plug-in bandwidth={h_opt}')
    leg5 = mpatches.Patch(color=None, label=f'cross-validated bandwidth={h_cv}')

    ax4.legend(handles=[leg4])
    ax5.legend(handles=[leg5])

    plt.tight_layout()
    plt.show()
