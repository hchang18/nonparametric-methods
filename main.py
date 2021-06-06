from kernel_density_estimator import plot_kde, gaussian_pdf
from util import *
from kernel_regress_estimator import *
from median_test import *
from symmetry_test import *
from exchangeability_test import *
from dispersion_test import *
from independence_test import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # kernel density estimation
    # plot_kde(true_dist='norm', num_samples=100, kernel_function=gaussian_pdf, bandwidth_h=[0.1, 0.4, 1])

    # kernel regression estimation
    # data, x, y = read_data("data1.txt")
    # plot_2darray(x, y)
    # plot_kre(data, gaussian_pdf)

    # nonparametric testing

    # wilcoxon_test
    n = 30
    data = []
    for i in range(n):
        data.append(np.random.normal(0.5, 1))
    w, p_value = wilcoxon_test(data)

    # mann_whitney_u_test
    # Generate X from N(0.2, 1) and Y from N(0.5, 1)
    # m = 300  # number of Xs
    # n = 400  # number of Ys
    # X = np.random.normal(0.2, 1, size=m)
    # Y = np.random.normal(0.5, 1, size=n)
    # mann_whitney_u_test(X, Y)

    # fligner_policello test
    # Generate X from N(0.2, 0.5) and Y from N(0.5, 1)
    # m = 300  # number of Xs
    # n = 400  # number of Ys
    # X = np.random.normal(0.2, 0.5, size=m)
    # Y = np.random.normal(0.5, 1, size=n)
    # fligner_policello_test(X, Y)

    # test symmetry
    # data, X, Y = read_data("data2.txt")
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.hist(X)
    # ax1.set_title("X")
    # ax2.hist(Y)
    # ax2.set_title("Y")
    # fig.show()
    # z, pvalue = symmetry_test(X)
    # z1, pvalue1 = symmetry_test(Y)
    # print(z, pvalue)
    # print(z1, pvalue1)

    # test exchangeability
    # n = 20  # samples or trials
    # p_x = 0.4
    # p_y = 0.5
    # size = 15  # number of experiments
    # x = np.random.binomial(n, p_x, size)
    # y = np.random.binomial(n, p_y, size)
    # exchangeability_test(x, y, size)

    # test whether variances are the same
    # data, X, Y = read_data("data2.txt")
    # Q, p_value = variance_test_one(X, Y)
    # if Q > 1.95 or Q < - 1.95:
    #     print("Reject H0 (variances of two distributions are the same)")
    # else:
    #     print("Cannot reject H0 (variances of two distributions are the same)")
    # variance_test_two(X, Y)


    # test independence
    # data, X, Y = read_data("data2.txt")
    # independence_test(data)

