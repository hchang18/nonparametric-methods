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
# returns test statistic and p-value
symmetry_test(X)

# test if the median of X and Y are the same
# both tests return test statistic and p-value
# mann whitney test
mann_whitney_u_test(X, Y)
# fligner-policello test
fligner_policello_test(X, Y)

# test if the variance of X and Y are the same
# returns test statistic and p-value
variance_test_one(X, Y)
variance_test_two(X, Y)

# test the independence between X and Y
# returns test statistic, p-value, kendall's tau
independence_test(data)

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
# plot the relationship between two samples
# using the kernel regression estimator
bandwidth = 'crossval'
plot_kre(data, gaussian_pdf, bandwidth)
plt.show()
```

Refer to `main.py` for complete demo. 


## Contributing
Pull requests are welcome. 
For major changes, please open an issue first to discuss 
what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)
