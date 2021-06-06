# median_test.py
import numpy as np
from scipy.stats import norm


def wilcoxon_test(data):
    """
    Calculate the Wilcoxon signed-rank test

    The Wilcoxon signed-rank tests the null hypothesis that
    two related paired samples come from the same distribution.
    It tests whether the distribution of the difference x - y
    is symmetric about zero.

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
    print(n)
    absolute_values = []
    for d in data:
        absolute_values.append((d, np.abs(d)))

    absolute_values.sort(key=lambda x: x[1])
    ret = []
    for i, d in enumerate(absolute_values):
        ret.append((i + 1, d[0], d[1]))

    t_plus = 0
    t_minus = 0
    for tup in ret:
        if tup[1] < 0:
            t_minus += tup[0]
        else:
            t_plus += tup[0]

    w = min(t_plus, t_minus)

    E_w = n * (n + 1) / 4
    se = np.sqrt(n * (n+1) * (2*n+1)/24)
    z = (w - E_w) / se
    p_value = 2. * norm.sf(abs(z)) # two sided test

    return z, p_value


def mann_whitney_u_test(X, Y):
    """
    Calculate the Mann-Whitney rank test on samples X and Y.
    It tests whether they have the same median.

    Parameters
    ----------
    X : array of floats
    Y : array of floats

    Returns
    -------
    statistic : float
    p_value : float
        THe p-value for the two-sided test
    """
    m, n = len(X), len(Y)

    U = 0
    for x in X:
        for y in Y:
            if x < y:
                U += 1

    E_u = m * n / 2
    var_u = m * n * (m + n + 1) / 12

    z = (U - E_u) / np.sqrt(var_u)
    p_value = 2. * norm.sf(abs(z)) # two sided test

    return z, p_value


def fligner_policello_test(X, Y):
    """
    Calculate the Fligner-Policello test on samples X and Y.
    It tests whether they have the same median, but without
    assumption on shape or scale of the distributions. However,
    it assumes that X and Y are from two different symmetric
    distributions.

    Parameters
    ----------
    X : array of floats
    Y : array of floats

    Returns
    -------
    statistic : float
    p_value : float
        THe p-value for the two-sided test
    """
    P_i = []
    for x in X:
        count = 0
        for y in Y:
            if y <= x:
                count += 1
        P_i.append(count)

    Q_j = []
    for y in Y:
        count = 0
        for x in X:
            if x <= y:
                count += 1
        Q_j.append(count)

    P_i = np.array(P_i)
    Q_j = np.array(Q_j)
    P_bar = np.average(P_i)
    Q_bar = np.average(Q_j)
    V1 = sum((P_i - P_bar) ** 2)
    V2 = sum((Q_j - Q_bar) ** 2)
    z = (sum(Q_j) - sum(P_i)) / (2 * np.sqrt(V1 + V2 + P_bar * Q_bar))
    p_value = 2. * norm.sf(abs(z)) # two sided test

    return z, p_value
