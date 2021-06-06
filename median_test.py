# median_test.py
import numpy as np


def wilcoxon_test(data):
    abs = []
    for d in data:
        abs.append((d, np.abs(d)))

    a = sorted(abs, key=lambda x: x[1])
    ret = []
    for i, d in enumerate(a):
        ret.append((i + 1, d[0], d[1]))

    t_plus = 0
    t_minus = 0
    for tup in ret:
        if tup[1] < 0:
            t_minus += tup[0]
        else:
            t_plus += tup[0]

    w = min(t_plus, t_minus)

    # test at 5% confidence level
    # reject if min < critical value
    # 55 is smaller than critical value 137 for n = 30
    if w < 137:
        print("reject H0 that median is 0")
    else:
        print("cannot reject H0")

    return w


def mann_whitney_u_test(X, Y):
    m, n = len(X), len(Y)

    U = 0
    for x in X:
        for y in Y:
            if x < y:
                U += 1

    E_u = m * n / 2
    var_u = m * n * (m + n + 1) / 12

    z = (U - E_u) / np.sqrt(var_u)

    if z > 1.95 or z < - 1.95:
        print("Reject H0 (medians of two distributions are the same)")
    else:
        print("Cannot reject H0 (medians of two distributions are the same)")


def fligner_policello_test(X, Y):
    m, n = len(X), len(Y)

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
    U_hat = (sum(Q_j) - sum(P_i)) / (2 * np.sqrt(V1 + V2 + P_bar * Q_bar))

    if U_hat > 1.95 or U_hat < - 1.95:
        print("Reject H0 (medians of two distributions are the same)")
    else:
        print("Cannot reject H0 (medians of two distributions are the same)")