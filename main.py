import seaborn as sns
from util import *
from kernel_regress_estimator import *
from median_test import *
from symmetry_test import *
from dispersion_test import *
from independence_test import *


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # import dataset
    data, X, Y = read_data("data2.txt")

    # provide fundamental summaries of two samples
    sns.set(color_codes=True)
    plt.rcParams["figure.figsize"] = (10, 7.5)
    plt.rcParams["axes.titlesize"] = 20

    # Histogram of X
    fig, ax = plt.subplots()
    ax.hist(X)
    ax.grid(True)
    ax.set_title('Histogram of X')
    fig.show()

    # Histogram of Y
    fig, ax = plt.subplots()
    ax.hist(Y)
    ax.grid(True)
    ax.set_title('Histogram of Y')
    fig.show()

    # test of symmetry - X
    X_v, X_pvalue = symmetry_test(X)
    print(f"Symmetry test (X): p-value is {X_pvalue}")

    Y_v, Y_pvalue = symmetry_test(Y)
    print(f"Symmetry test (Y): p-value is {Y_pvalue}")

    diff, diff_pvalue = symmetry_test(X-Y)
    print(f"symmetry test (X-Y): p-value is {diff_pvalue}")

    # test if the median of X and Y are the same
    # mann whitney
    m, m_pvalue = mann_whitney_u_test(X, Y)
    print(f"Mann Whitney U test: p-value is {m_pvalue}")
    # fligner-policello
    fp, fp_pvalue = fligner_policello_test(X, Y)
    print(f"fligner-policello test: p-value is {fp_pvalue}")

    # test if the variance of X and Y are the same
    c, q1_pvalue = variance_test_one(X, Y)
    print(f"variance test one: Q is {c} and p-value is {q1_pvalue}")
    q, q2_pvalue = variance_test_two(X, Y)
    print(f"variance test two: Q is {q} and p-value is {q2_pvalue}")

    # test the independence between X and Y
    z_ind, p_val_ind, tau = independence_test(data)
    print(f"Independence test: z score is {z_ind}, p-value is {p_val_ind}, and tau is {tau}")

    # since we know that they are dependent
    # let's find out the relationship
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Scatter Plot of X and Y')
    ax.scatter(X, Y)
    ax.grid(True)
    leg = mpatches.Patch(color=None, label='original data plots')
    ax.legend(handles=[leg])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.show()

    # plot the relationship between two samples
    # using the kernel regression estimator
    bandwidth = 'crossval'
    plot_kre(data, gaussian_pdf, bandwidth)


