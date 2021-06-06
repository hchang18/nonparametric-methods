# symmetry_test
from itertools import combinations
import numpy as np


# test symmetry of the sample
def symmetry_test(a):
    # Generate triples of observations
    # A Python program to print all
    # combinations of given length
    # Get all combinations of generated data and length 3
    comb = combinations(a, 3)

    def calculate_F_star(triple):
        return np.sign(triple[0] + triple[1] - 2 * triple[2]) + np.sign(
            triple[1] + triple[2] - 2 * triple[0]) + np.sign(triple[0] + triple[2] - 2 * triple[1])

    T = 0
    f_star = []
    # Print the obtained combinations
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
    # print(list(comb))

    for tupl in list(comb):
        s, t = tupl[0], tupl[1]
        # print(s, t)
        first = 0
        for j in range(0, s):
            first += calculate_F_star((a[j], a[s], a[t]))

        second = 0
        for j in range(s + 1, t):
            second += calculate_F_star((a[s], a[j], a[t]))

        third = 0
        for j in range(t + 1, n):
            third += calculate_F_star((a[s], a[t], a[j]))

        b_st = first + second + third
        B_st.append(b_st)

    sigma_squared = ((n - 3) * (n - 4)) / ((n - 1) * (n - 2)) * sum([x * x for x in B_t]) + (n - 3) / (n - 4) * sum(
            [x * x for x in B_st]) + n * (n - 1) * (n - 2) / 6 - (
                                1 - ((n - 3) * (n - 4) * (n - 5)) / (n * (n - 1) * (n - 2))) * T ** 2

    sigma = np.sqrt(sigma_squared)
    v = T / sigma

    # test at 5% confidence level
    print("Symmetric test")
    if np.abs(v) > 1.96:
        print("reject H0 that F is symmetric")
    else:
        print("cannot reject H0")
