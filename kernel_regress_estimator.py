# kernel_regress_estimator.py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import norm, expon
from random import seed, randrange, uniform

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

plt.rcParams["figure.figsize"] = (15, 10)


# ============================================
# Kernel Regression Estimator                |
# ============================================
def kernel_regression_estimator(data, kernel_func, bandwidth):
    """ Generate kernel regression estimator over data."""
    X = data[:, 1]
    Y = data[:, 0]
    kernels = dict()
    n = len(X)
    for d in X:
        kernels[d] = kernel_func(d, bandwidth)

    def evaluate(x):
        """Evaluate `x` using kernels above."""
        resp = list()
        weight = list()
        for d in X:
            resp.append(kernels[d](x))
        resp_sum = sum(resp)  # denominator
        for i in range(n):
            weight.append(resp[i] / resp_sum)
        result = list()
        for i in range(n):
            result.append(weight[i] * Y[i])
        return sum(result)

    return (evaluate)


# ============================================
# Gaussian Kernel PDF                        |
# ============================================
def gaussian_pdf(x_i, bandwidth):
    """Return Gaussian kernel density estimator."""
    x_bar = x_i

    def evaluate(x):
        """Evaluate x."""
        pdf = (np.sqrt(2 * np.pi * bandwidth ** 2) ** -1) * np.exp(-((x - x_bar) ** 2) / (2 * bandwidth ** 2))
        return (pdf)

    return (evaluate)


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


def estimate_bandwidth(data, kernel_function):
    num_samples = len(data[:, 0])
    y = data[:, 0]
    x = data[:, 1]
    # list of bandwidth
    bandwidths = np.arange(0.01, 2, 0.02)

    # estimate y_hat corresponding to X
    errors = list()
    k = 10
    for h in bandwidths:
        error = 0
        folds = cross_validation_split(data, k)
        for i in range(k):
            test, train = bundle_test_train_set(data, k, i)
            estimator = kernel_regression_estimator(train, kernel_func=kernel_function, bandwidth=h)
            y_hat = [estimator(x) for x in test[:, 1]]
            error += (test[:, 0] - y_hat) ** 2
        errors.append(sum(error))

    errors = np.array(errors)
    h_opt = bandwidths[np.argmin(errors)]
    return h_opt


# =========================================
# kernel density estimates visualizations |
# =========================================
def plot_kre(data, kernel_function):
    seed(2)

    num_samples = len(data[:, 0])
    vals = data[:, 1]  # x
    xvals = np.arange(min(vals), max(vals), .01)

    # ========================================================
    # Bandwidth Selection : rule-of-thumb plugin             |
    # ========================================================
    # bandwidth estimation based on kernel function

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
    h_cv = estimate_bandwidth(data, gaussian_pdf)

    # ========================================================
    # Optimized Bandwidth visualization                      |
    # ========================================================
    fig = plt.figure()

    # bandwidth=optimal_bandwidth_plugin:
    ax4 = fig.add_subplot(2, 2, 1)
    dist_4 = kernel_regression_estimator(data, kernel_func=kernel_function, bandwidth=h_opt)
    y4 = [dist_4(i) for i in xvals]
    ax4.scatter(data[:, 1], data[:, 0])
    ax4.plot(xvals, y4)

    # bandwidth=optimal_bandwidth_crossvalidated:
    ax5 = fig.add_subplot(2, 2, 2)
    dist_5 = kernel_regression_estimator(data, kernel_func=kernel_function, bandwidth=h_cv)
    y5 = [dist_5(i) for i in xvals]
    ax5.scatter(data[:, 1], data[:, 0])
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


# ===================================================
# Generate data                                     |
# ===================================================
def generate_data(n):
    x = list()
    y = list()

    for i in range(n):
        rand_x = uniform(-2, 10)
        x.append(rand_x)
        if rand_x < 1:
            rand_y = (rand_x ** 2) / 20
        elif 1 <= rand_x < 4:
            rand_y = rand_x / 10 - 1 / 20
        else:
            rand_y = rand_x * np.sin(rand_x - 2.4) / 16 - 0.5 + 7 / 20

        eta = np.random.normal(0, 0.2)
        y.append(rand_y + eta)

    x = np.array(x).reshape(n, 1)
    y = np.array(y).reshape(n, 1)
    data = np.hstack((np.array(y), np.array(x)))

    return data, y, x


def create_confidence_interval(num_simulation):
    # repeat the estimation 1000 times
    f_hat_list = []

    for i in range(num_simulation):
        # generate data
        n = 200
        data, _, _ = generate_data(n)
        xi = data[:, 1]
        x = np.arange(-2, 10, .1)
        # estimate bandwidth - calculate on the first dataset
        if i == 0:
            h_cv = estimate_bandwidth(data, gaussian_pdf)
            print(h_cv)
        # estimate f_hat
        estimator = kernel_regression_estimator(data, gaussian_pdf, bandwidth=h_cv)
        y_hat = [estimator(i) for i in x]

        f_hat_list.append(y_hat)
