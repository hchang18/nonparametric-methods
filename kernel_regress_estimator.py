# kernel_regress_estimator.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from random import uniform
from kernel_density_estimator import gaussian_pdf, bundle_test_train_set, calculate_optimum_bandwidth

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

    return evaluate


def estimate_bandwidth(data, kernel_function):
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


# =========================================
# kernel density estimates visualizations |
# =========================================
def plot_kre(data, kernel_function):
    x_values = data[:, 1]  # x
    x = np.arange(min(x_values), max(x_values), .01)

    h_opt = calculate_optimum_bandwidth(x_values, kernel_function)

    # ========================================================
    # Bandwidth Selection : cross-validation                 |
    # ========================================================
    h_cv = estimate_bandwidth(data, gaussian_pdf)

    # ========================================================
    # Optimized Bandwidth visualization                      |
    # ========================================================
    fig = plt.figure()
    # plugin optimal bandwidth
    ax = fig.add_subplot(2, 2, 1)
    dist_h_opt = kernel_regression_estimator(data, kernel_func=kernel_function, bandwidth=h_opt)
    y_h_opt = [dist_h_opt(i) for i in x]
    ax.scatter(data[:, 1], data[:, 0])
    ax.plot(x, y_h_opt)

    # bandwidth chosen from cross validation
    ax1 = fig.add_subplot(2, 2, 2)
    dist_h_cv = kernel_regression_estimator(data, kernel_func=kernel_function, bandwidth=h_cv)
    y_h_cv = [dist_h_cv(i) for i in x]
    ax1.scatter(data[:, 1], data[:, 0])
    ax1.plot(x, y_h_cv)

    # display gridlines
    ax.grid(True)
    ax1.grid(True)

    # display legend in each subplot
    leg4 = mpatches.Patch(color=None, label=f'plug-in bandwidth={h_opt}')
    leg5 = mpatches.Patch(color=None, label=f'cross-validated bandwidth={h_cv}')

    ax.legend(handles=[leg4])
    ax1.legend(handles=[leg5])

    plt.tight_layout()
    plt.show()


# ===================================================
# Generate data                                     |
# ===================================================
def generate_data_for_midterm(n):
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
        data, _, _ = generate_data_for_midterm(n)
        x = np.arange(-2, 10, .1)
        # estimate bandwidth - calculate on the first dataset
        if i == 0:
            h_cv = estimate_bandwidth(data, gaussian_pdf)
            print(h_cv)
        # estimate f_hat
        estimator = kernel_regression_estimator(data, gaussian_pdf, bandwidth=h_cv)
        y_hat = [estimator(i) for i in x]

        f_hat_list.append(y_hat)

    return f_hat_list
