# dispersion_test
import numpy as np


def variance_test_one(X, Y):
    # Pool in one
    m = len(X)
    n = len(Y)
    X_set = set(X)
    Y_set = set(Y)
    S = np.union1d(X, Y)
    N = m + n

    # N is odd
    if N % 2 == 1:
        print("N odd")
        # create the grid
        grid = [i + 1 for i in range(int(N / 2))]
        reversed_grid = grid[::-1]
        middle = [int((N + 1) / 2)]
        R = grid + middle + reversed_grid
        S.sort()

        # calculate C
        C = 0
        for k in range(N):
            if S[k] in Y_set:
                C += R[k]

        E_c = (n * (N + 1) ** 2) / (4 * N)
        var_c = n * (N - n) * (N + 1) * (N ** 2 + 3) / (48 * N ** 2)

        Q = (C - E_c) / np.sqrt(var_c)
    else:
        print("N even")

        grid = [i + 1 for i in range(int(N / 2))]
        reversed_grid = grid[::-1]
        R = grid + reversed_grid
        S.sort()

        # calculate C
        C = 0
        for i in range(N):
            if S[i] in Y_set:
                C += R[i]

        E_c = n * (N + 2) / 4
        var_c = n * (N - n) * (N + 2) * (N - 2) / (48 * (N - 1))

        Q = (C - E_c) / np.sqrt(var_c)

    if Q > 1.95 or Q < - 1.95:
        print("Reject H0 (variances of two distributions are the same)")
    else:
        print("Cannot reject H0 (variances of two distributions are the same)")


def variance_test_two(X, Y):
    m = len(X)
    n = len(Y)

    X_sum = np.full(m, sum(X))
    X_s = X_sum - X
    X_i_bar = X_s / (m - 1)
    D_i = (X_s - X_i_bar) ** 2 / (m - 2)

    Y_sum = np.full(n, sum(Y))
    Y_s = Y_sum - Y
    Y_j_bar = Y_s / (n - 1)
    E_j = (Y_s - Y_j_bar) ** 2 / (n - 2)

    X_0_bar = sum(X) / m
    D_0 = sum((X - X_0_bar) ** 2 / (m - 1))

    Y_0_bar = sum(Y) / n
    E_0 = sum((Y - Y_0_bar) ** 2 / (n - 1))

    S_i = np.log(D_i)
    T_j = np.log(E_j)
    S_0 = np.log(D_0)
    T_0 = np.log(E_0)

    A_i = m * S_0 - (m - 1) * S_i
    B_j = n * T_0 - (n - 1) * T_j

    A_bar = sum(A_i) / m
    B_bar = sum(B_j) / n

    V1 = 1 / (m * (m - 1)) * sum((A_i - A_bar) ** 2)
    V2 = 1 / (n * (n - 1)) * sum((B_j - B_bar) ** 2)

    Q = (A_bar - B_bar) / np.sqrt(V1 + V2)
    print(f"Q: {Q}")

    if Q > 1.95 or Q < - 1.95:
        print("Reject H0 (variances of two distributions are the same)")
    else:
        print("Cannot reject H0 (variances of two distributions are the same)")