from demos.setup import demo, np, plt
from compecon import BasisChebyshev
get_ipython().magic('matplotlib inline')

f = lambda x: 1 + np.sin(2 * x)

n, a, b = 5, 0, np.pi
B = BasisChebyshev(n, a, b, f=f)

xnodes = B.nodes
print(xnodes)

B.plot()

# c = np.mat(np.linalg.solve(B.Phi(), f(xnodes)))  
# notice that Phi is a dictionary, the matrix is indexed by 0

x = np.linspace(a, b, 121)
f_approx = B(x)
f_true = f(x)

plt.figure()
demo.subplot(2, 1, 1, 'Chebyshev approximation with 5 nodes', '', '$f(x)$', [a, b])
plt.plot(x, np.c_[f_true, f_approx])
plt.legend(['$f = 1 + sin(2x)$', 'approx.'], loc=3)

demo.subplot(2, 1, 2, '', '$x$', 'Residuals', [a, b])
plt.plot(x.T, f_approx - f_true)

B = BasisChebyshev(25, a, b, f=f)
plt.figure()
demo.subplot(2, 1, 1, 'Chebyshev approximation with 25 nodes', '', '$f(x)$', [a, b])
plt.plot(x, np.c_[f_true, B(x)])
plt.legend(['$f = 1 + sin(2x)$', 'approx.'], loc=3)

demo.subplot(2, 1, 2, '', '$x$', 'Residuals', [a, b])
plt.plot(x.T, B(x) - f_true)

df = lambda x: 2 * np.cos(2 * x)
plt.figure()
demo.subplot(2, 1, 1, 'Approximating the derivative, 25 nodes', '', "$f'(x)$", [a, b])
plt.plot(x, np.c_[df(x), B(x, 1)])  # notice the 1 in B(x, 1) to compute first derivative
plt.legend(['df/dx = 2cos(2x)', 'approx.'], loc=3)

demo.subplot(2, 1, 2, '', '$x$', 'Residuals', [a, b])
plt.plot(x, B(x, 1) - df(x))

