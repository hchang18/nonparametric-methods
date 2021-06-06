# independence_test
import numpy as np


def independence_test(data):

    n = len(data)
    # sort the array with X
    sorted_data = np.array(sorted(data, key=lambda x: x[1]))
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
    print(f'tau: {tau}')

    # source of z calculation: https://www.statology.org/kendalls-tau/
    # statistical significance of Kendall's tau
    # since n = 200, tau generally follows normal distribution
    z = 3 * tau * np.sqrt(n * (n - 1)) / np.sqrt(2 * (2 * n + 5))

    if (z > 1.95 or z < - 1.95):
        print("Reject H0 (two samples are independent)")
    else:
        print("Cannot reject H0 (two samples are independent)")

    def required_sample_size():
        # optimal sample size
        z_alpha = 1.96
        z_beta = 1.96
        min_n = 4 * (z_alpha + z_beta) ** 2 / (9 * tau ** 2)
        return int(min_n + 1)

    print(f'minimum required sample size: {required_sample_size()}')

