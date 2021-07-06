# Nonparametric method library

by Haeyoon Chang 2021 based on lecture notes by Prof. Ge Zhao @ Portland State University 

Nonparametric method is a type of statistic that does not make any 
assumption on characteristics or parameters of the samples. This repo contains 
nonparametric estimators such as kernel density estimator and kernel regression estimator, 
and nonparametric testings including median test, symmetry test, dispersion test, independence test.   

## Code 

- kernel_density_estimator.py 
    - kde_pdf
    - uniform_pdf
    - epanechnikov_pdf
    - gaussian_pdf
    - calculate_optimum_bandwidth
    - plot_kde
- kernel_regress_estimator.py
    - kernel_regression_estimator
    - estimate_bandwidth
    - plot_kre
- dispersion_test.py
- exchangeability_test.py
- independence_test.py
- median_test.py
    - wilcoxon_test
    - mann_whitney_u_test
    - fligner_policello_test
- symmetry_test.py

## Demo

```python
from kernel_regress_estimator import *
from median_test import *
from symmetry_test import *
from dispersion_test import *
from independence_test import *

# test of symmetry of distribution of data X
X_v, X_pvalue = symmetry_test(X)

# test if the median of X and Y are the same
# mann whitney test
m, m_pvalue = mann_whitney_u_test(X, Y)
# fligner-policello test
fp, fp_pvalue = fligner_policello_test(X, Y)

# test if the variance of X and Y are the same
c, q1_pvalue = variance_test_one(X, Y)
q, q2_pvalue = variance_test_two(X, Y)

# test the independence between X and Y
z_ind, p_val_ind, tau = independence_test(data)

# when two datasets are dependent, 
# we can find out the relationship between X and Y
# using kernel regression estimator
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Scatter Plot of X and Y')
ax.scatter(X, Y)
ax.grid(True)
leg = mpatches.Patch(label='original data plots')
ax.legend(handles=[leg])
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
plt.show()

# plot the relationship between two samples
# using the kernel regression estimator
bandwidth = 'crossval'
plot_kre(data, gaussian_pdf, bandwidth)
```

Refer to `main.py` for complete demo. 


## Contributing
Pull requests are welcome. 
For major changes, please open an issue first to discuss 
what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)