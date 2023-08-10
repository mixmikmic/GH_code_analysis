import numpy as np
import scipy.optimize

def l1_fit(U, v):
    """
    Find a least absolute error solution (m, k) to U * m + k = v + e.
    Minimize sum of absolute values of vector e (the residuals).
    """
    U = np.array(U)
    v = np.array(v)
    # n is the number of samples
    n = len(v)
    s = U.shape
    assert len(s) == 2
    assert s[0] == n
    # d is the number of dimensions
    d = s[1]
    I = np.identity(n)
    n1 = np.ones((n,1))
    A = np.vstack([
            np.hstack([-I, U, n1]),
            np.hstack([-I, -U, -n1])
        ])
    c = np.hstack([np.ones(n), np.zeros(d+1)])
    b = np.hstack([v, -v])
    bounds = [(0, None)] * n + [(None, None)] * (d+1)
    options = {"maxiter": 10000}
    # Call the linprog subroutine.
    r = scipy.optimize.linprog(c, A, b, bounds=bounds, options=options)
    # Extract the interpolation result from the linear program solution.
    x = r.x
    m = x[n:n+d]
    k = x[n+d]
    v_predicted = np.dot(U, m) + k
    residuals = v - v_predicted
    # For debugging store all parameters, intermediates and results in returned dict.
    result = {}
    result["U"] = U
    result["v"] = v
    result["m"] = m
    result["k"] = k
    result["r"] = r
    result["samples"] = n
    result["dimensions"] = d
    result["A"] = A
    result["b"] = b
    result["c"] = c
    result["bounds"] = bounds
    result["residuals"] = residuals
    result["v_predicted"] = v_predicted
    return result

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


def lsqfit(x, y):
    "least squares fit, for comparison"
    x = np.array(x)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    fity = m*x + c
    return fity

def plot(result, size=64):
    samples = result["samples"]
    dimensions = result["dimensions"]
    # this plot only works for linear fits
    assert dimensions == 1, "sorry, no can do"
    U = result["U"]
    m = result["m"]
    x = U.reshape((samples,))
    y_actual = result["v"]
    y_predicted = result["v_predicted"]
    y_least_squares = lsqfit(x, y_actual)
    fig, ax = plt.subplots()
    ax.scatter(x, y_actual, facecolors='none', edgecolors='b', s=size)
    #ax.scatter(x, y_predicted, color="red", marker="x", s=size)
    ax.plot(x, y_least_squares, color="green", dashes=(5,5))
    ax.plot(x, y_predicted, color="red")
    fig.show()

# Let's make the numbers more easy to read...
get_ipython().run_line_magic('precision', '3')

# Trivial example usage
# Independent variable values.
U = ([1], [2])
# Dependent variable values
v = (0, 1)
# Perform the regression.
result = l1_fit(U, v)
result["m"], result["k"], result["residuals"], result["samples"], result["dimensions"]

plot(result)

# Less trivial example usage
# Independent variable values
U = ([1], [2], [3])
# Dependent variable values
v = (0, 0, 1)
result = l1_fit(U, v)
plot(result)
result["m"], result["k"], result["residuals"], result["samples"], result["dimensions"]

# A trivial example with 2 predictor variables x + y - 2
U = ([1, 0], [2, 1], [3, 0])
v = (-1, 1, 1)
result = l1_fit(U, v)
result["m"], result["k"], result["residuals"], result["samples"], result["dimensions"]

# A non trivial example with 2 predictor variables x + y - 2 and one outlier
U = ([1, 1], [1, -1], [-1, -1], [-1, 1], [0, 0])
v = (0, -2, -4, -2, -1.3)
result = l1_fit(U, v)
result["m"], result["k"], result["residuals"], result["samples"], result["dimensions"]

# nearly linear with an outlier
n = 10
x = np.linspace(0,1,n)
small_error = (np.random.randn(n)-0.5) * 0.1
y = (0.5 * x - 1) + small_error
y[5] = 5 # outlier!

U = x.reshape((n, 1))
v = y
result = l1_fit(U, v)
plot(result)

# a random 1d example
n = 20
x = np.array(sorted(np.random.randn(n)))
y = np.random.randn(n)
U = x.reshape((n, 1))
v = y
result = l1_fit(U, v)
plot(result)



