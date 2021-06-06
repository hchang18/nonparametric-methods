# exchangeability_test
import numpy as np


def exchangeability_test(x, y, size):
    ab = []
    for i in range(size):
        ab.append((x[i], y[i], min(x[i], y[i]), max(x[i], y[i])))  # x, y, a, b

    # sort it by a
    ab.sort(key=lambda x: x[2])

    r = []
    for i in range(size):
        if ab[i][0] == ab[i][2]:
            r.append(1)
        else:
            r.append(-1)

    d = []
    T_j = []
    for j in range(size):
        d_j = []
        for i in range(size):
            if ab[j][3] >= ab[i][3] > ab[j][2] >= ab[i][2]:
                d_j.append(1)
            else:
                d_j.append(0)
        # print("r: ", r)
        # print("d_j: ", d_j)
        t = [x * y for x, y in zip(r, d_j)]
        # print("t: ", t)
        T_j.append(sum(t))
        d.append(d_j)

    # print(np.array(d).shape)
    A_obs = sum([t ** 2 for t in T_j]) / (size ** 2)
    print(A_obs)

    r_n = []
    possible_outcomes = 2 ** size
    for i in range(possible_outcomes):
        binary = bin(i)[2:].zfill(size)
        r = list(map(int, binary))
        r = [-1 if x == 0 else x for x in r]
        r_n.append(r)

    A = []
    T = []
    for i in range(possible_outcomes):
        for j in range(size):
            t = [x * y for x, y in zip(r_n[i], d[j])]
            T.append(sum(t))
        A.append(sum([t ** 2 for t in T]) / (size ** 2))

    alpha = 0.95
    m = 2 ** size - np.floor(2 ** size * alpha)
    print(A[int(m)])

    # test at 5% confidence level
    # testing exchangeability
    print("exchangeability test")
    if A_obs > A[int(m)]:
        print("reject H0 that x and y are exchangeable")
    else:
        print("cannot reject H0")
