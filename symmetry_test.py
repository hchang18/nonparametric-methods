# symmetry_test.py
from itertools import combinations
import numpy as np
from scipy.stats import norm


def symmetry_test(a):
    """
    Calculate symmetry test
    It tests whether the given probability density distribution is
    symmetric around the median.

    Parameters
    ----------
    a : array of floats

    Returns
    -------
    statistic : float
    p_value : float
        THe p-value for the two-sided test
    """
    # Get all combinations of generated data and length 3
    comb = combinations(a, 3)

    def calculate_F_star(triple):
        return np.sign(triple[0] + triple[1] - 2 * triple[2]) + np.sign(
            triple[1] + triple[2] - 2 * triple[0]) + np.sign(triple[0] + triple[2] - 2 * triple[1])

    f_star = []
    for triple in list(comb):
        f = calculate_F_star(triple)
        f_star.append(f)

    T = sum(f_star)

    n = len(a)

    B_t = []
    for t in range(n):
        first = 0
        for j in range(t + 1, n - 1):
            for k in range(j + 1, n):
                first += calculate_F_star((a[t], a[j], a[k]))

        second = 0
        for j in range(0, t):
            for k in range(t + 1, n):
                second += calculate_F_star((a[j], a[t], a[k]))

        third = 0
        for j in range(0, t - 1):
            for k in range(j + 1, t):
                third += calculate_F_star((a[j], a[k], a[t]))

        b_t = first + second + third
        B_t.append(b_t)

    B_st = []
    idx = [i for i in range(n)]
    comb = combinations(idx, 2)

    for tupl in list(comb):
        s, t = tupl[0], tupl[1]
        first, second, third = 0, 0, 0
        for j in range(0, s):
            first += calculate_F_star((a[j], a[s], a[t]))
        for j in range(s + 1, t):
            second += calculate_F_star((a[s], a[j], a[t]))
        for j in range(t + 1, n):
            third += calculate_F_star((a[s], a[t], a[j]))

        b_st = first + second + third
        B_st.append(b_st)

    sigma_squared = ((n - 3) * (n - 4)) / ((n - 1) * (n - 2)) * sum([x * x for x in B_t]) \
                    + (n - 3) / (n - 4) * sum([x * x for x in B_st]) \
                    + n * (n - 1) * (n - 2) / 6 \
                    - (1 - ((n - 3) * (n - 4) * (n - 5)) / (n * (n - 1) * (n - 2))) * T ** 2

    sigma = np.sqrt(sigma_squared)
    v = T / sigma
    p_value = 2. * norm.sf(abs(v)) # two sided test

    return v, p_value
