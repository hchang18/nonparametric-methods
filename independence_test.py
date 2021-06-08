# independence_test.py
import numpy as np
from scipy.stats import norm


def independence_test(data):
    """
    Calculate independence test.
    This test tests the null hypothesis that two columns in data
    are independent or orthogonal to each other.

    Parameters
    ----------
    data : 2d array of floats

    Returns
    -------
    statistic : float
    p_value : float
        THe p-value for the two-sided test
    """

    n = len(data)


    # sort the array with X
    def Sort_Tuple(tup):
        return sorted(tup, key=lambda x: x[1])

    sorted_data = np.array(Sort_Tuple(data))
    X = sorted_data[:, 1]
    Y = sorted_data[:, 0]


    concordant = 0
    discordant = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            if Y[j] > X[i]:
                concordant += 1
            if Y[j] < Y[i]:
                discordant += 1

    tau = (concordant - discordant) / (concordant + discordant)

    z = 3 * tau * np.sqrt(n * (n - 1)) / np.sqrt(2 * (2 * n + 5))
    p_value = 2. * norm.sf(abs(z)) # two sided test

    return z, p_value, tau


def calculate_required_sample_size(data):
    # optimal sample size
    z_alpha = 1.96
    z_beta = 1.96
    _, _, tau = independence_test(data)
    min_n = 4 * (z_alpha + z_beta) ** 2 / (9 * tau ** 2)
    return int(min_n + 1)


