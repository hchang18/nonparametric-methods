# kernel_regress_estimator.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from random import uniform
from kernel_density_estimator import gaussian_pdf, bundle_test_train_set, calculate_optimum_bandwidth
import seaborn as sns

def kernel_regression_estimator(data, kernel_func, bandwidth):
    """
    Calculate a kernel regression estimator using the
    given kernel function. Kernel regression estimation is a way
    to estimate the relationship between two variables in
    a non-parametric way.

    Parameters
    ----------
    data : arrays of floats

    kernel_func: function
        kernel function that applies to
        the fixed window and put more weight on points
        closer to the point being evaluated.

    bandwidth: float
        estimator bandwidth calculated by plugging in
        or from cross validation method.

    Return
    -------
    evaluate : function that evaluates x using given kernel
    """
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

    return evaluate


def estimate_bandwidth(data, kernel_function):
    """Compute the estimator bandwidth with cross validation"""
    bandwidths = np.arange(0.01, 2, 0.02)

    # estimate y_hat corresponding to X
    errors = list()
    k = 10
    for h in bandwidths:
        error = 0
        for i in range(k):
            test, train = bundle_test_train_set(data, k, i)
            estimator = kernel_regression_estimator(train, kernel_func=kernel_function, bandwidth=h)
            y_hat = [estimator(x) for x in test[:, 1]]
            error += (test[:, 0] - y_hat) ** 2
        errors.append(sum(error))

    errors = np.array(errors)
    h_opt = bandwidths[np.argmin(errors)]
    return h_opt


def plot_kre(data, kernel_function, bandwidth):
    """
    Visualize kernel regression estimates using both bandwidth
    obtained from plugin method and cross validation
    """
    x_values = data[:, 1]
    x = np.arange(min(x_values), max(x_values), .01)

    if 'plugin' in bandwidth:
        h = calculate_optimum_bandwidth(x_values, kernel_function)
    elif 'crossval' in bandwidth:
        h = estimate_bandwidth(data, gaussian_pdf)

    fig = plt.figure()

    # draw graph
    ax = fig.add_subplot(1, 1, 1)
    sns.set(color_codes=True)
    plt.rcParams["figure.figsize"] = (10, 7.5)
    plt.rcParams["axes.titlesize"] = 20
    ax.set_title('Kernel Regression Estimator')
    dist = kernel_regression_estimator(data, kernel_func=kernel_function, bandwidth=h)
    y = [dist(i) for i in x]
    ax.scatter(data[:, 1], data[:, 0])
    ax.plot(x, y, color='orange')
    ax.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    h_round = round(h, 2)
    leg = mpatches.Patch(color=None, label=f'bandwidth={h_round}')
    ax.legend(handles=[leg])
    plt.tight_layout()
    plt.show()


def generate_data_for_midterm(n):
    """
    Returns generated data that follows certain
    functions given below.
    """
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
    """
    Repeat num_simulation number of experiments
    with kernel_regression_estimator above.
    Return a list of function estimates.
    """
    f_hat_list = []

    for i in range(num_simulation):
        # generate data
        n = 200
        data, _, _ = generate_data_for_midterm(n)
        x = np.arange(-2, 10, .1)
        # estimate bandwidth - calculate on the first dataset
        if i == 0:
            h_cv = estimate_bandwidth(data, gaussian_pdf)
            print(h_cv)
        # estimate f_hat
        estimator = kernel_regression_estimator(data, gaussian_pdf, "crossvalidation")
        y_hat = [estimator(i) for i in x]

        f_hat_list.append(y_hat)

    return f_hat_list
