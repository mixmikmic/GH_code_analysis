get_ipython().magic('matplotlib notebook')
import numpy
from matplotlib import pyplot

tests = []

@tests.append
class f_expx:
    def F(x):
        return numpy.exp(2*x)/(1+x**2)
    def f(x):
        return 2*numpy.exp(2*x)/(1+x**2) - numpy.exp(2*x)/(1+x**2)**2 * 2*x

@tests.append
class f_quadratic:
    def F(x):
        return x**3/3 - x**2 + 3*x - 1
    def f(x):
        return x**2 - 2*x + 3
    
@tests.append
class f_tanh:
    def F(x):
        return numpy.tanh(x)
    def f(x):
        return numpy.cosh(x)**(-2)
    
@tests.append
class f_logx2:
    def F(x):
        return numpy.log(x**2)
    def f(x):
        return 1/x**2 * 2*x

pyplot.style.use('ggplot')
pyplot.rcParams['figure.max_open_warning'] = False
pyplot.figure()
x = numpy.linspace(-2,2)
for t in tests:
    pyplot.plot(x, t.f(x), label=t.__name__)
pyplot.ylim(-10,10)
pyplot.legend(loc='upper left')

def fint_midpoint(f, a, b, n=20):
    dx = (b - a)/n     # Width of each interval
    x = numpy.linspace(a+dx/2, b-dx/2, n)
    return numpy.sum(f(x))*dx

for t in tests:
    a, b = -2, 2.01
    I = fint_midpoint(t.f, a, b, 51)
    print('{:12s}: I={: 10e} error={: 10e}'.
          format(t.__name__, I, I - (t.F(b) - t.F(a))))

# How fast does this method converge?

def plot_accuracy(fint, tests, ns, ref=[1,2], plot=pyplot.loglog):
    a, b = -2, 2.001
    ns = numpy.array(ns)
    pyplot.figure()
    for t in tests:
        Is = numpy.array([fint(t.f, a, b, n) for n in ns])
        Errors = numpy.abs(Is - (t.F(b) - t.F(a)))
        plot(ns, Errors, '*', label=t.__name__)
    for k in ref:
        plot(ns, 1/ns**k, label='$1/n^{:d}$'.format(k))
    pyplot.ylabel('Error')
    pyplot.xlabel('n')
    pyplot.legend(loc='lower left')
    
plot_accuracy(fint_midpoint, tests, range(2,30))

x = numpy.linspace(-1,2,4)
print(x[0:2])
ix = [0,2]
print(x[ix])
print(x[[0,-1]])

def fint_trapezoid(f, a, b, n=20):
    dx = (b - a) / (n - 1)     # We evaluate endpoints of n-1 intervals
    x = numpy.linspace(a, b, n)
    fx = f(x)
    fx[[0,-1]] *= .5
    # fx[0] *= .5; fx[-1] *= .5
    return numpy.sum(fx)*dx

plot_accuracy(fint_trapezoid, tests, range(2,30))

for t in tests:
    a, b = -2, 2
    npoints = 10*2**numpy.arange(5)
    for n in npoints:
        I = fint_trapezoid(t.f, a, b, n)
        print('{:12s}: n={: 4d} I={: 10f} error={: 10f}'.
              format(t.__name__, n, I, I - (t.F(b) - t.F(a))))



pyplot.figure()
for t in tests[:-1]:
    h = 1/(npoints - 1)
    pyplot.loglog(h, [numpy.abs(fint_trapezoid(t.f, a, b, n) - (t.F(b) - t.F(a)))
                        for n in npoints], '*', label=t.__name__)
pyplot.loglog(h, h**2, label='$h^2$')
pyplot.xlabel('h')
pyplot.ylabel('I')
pyplot.legend(loc='upper left')

for t in tests[:-1]:
    a, b = -2, 2
    for n in [10, 20, 40, 80]:
        I_h = fint_trapezoid(t.f, a, b, n)
        I_2h = fint_trapezoid(t.f, a, b, n//2)
        I_extrap = I_h + (I_h - I_2h) / 3
        I_exact = t.F(b) - t.F(a)
        print('{:12s}: n={: 4d} error={: 10f} {: 10f} {: 10f}'.
              format(t.__name__, n, I_h-I_exact, I_2h-I_exact, I_extrap-I_exact))

@tests.append
class f_sin10:
    def F(x):
        return numpy.sin(10*x)
    def f(x):
        return 10*numpy.cos(10*x)

def fint_richardson(f, a, b, n):
    n = (n // 2) * 2 + 1
    h = (b - a) / (n - 1)
    x = numpy.linspace(a, b, n)
    fx = f(x)
    fx[[0,-1]] *= .5
    I_h = numpy.sum(fx)*h
    I_2h = numpy.sum(fx[::2])*2*h
    return I_h + (I_h - I_2h) / 3

plot_accuracy(fint_richardson, tests, range(3,60,2), ref=[2,3,4])

def vander_legendre(x, n=None):
    if n is None:
        n = len(x)
    P = numpy.ones((len(x), n))
    if n > 1:
        P[:,1] = x
    for k in range(1,n-1):
        P[:,k+1] = ((2*k+1) * x * P[:,k] - k * P[:,k-1]) / (k + 1)
    return P

def fint_legendre_lin(f, a, b, n):
    x = numpy.linspace(a, b, n)
    fx = f(x)
    P = vander_legendre(numpy.linspace(-1,1,n))
    c = numpy.linalg.solve(P, fx)
    return c[0] * (b-a)

plot_accuracy(fint_legendre_lin, tests[:-1], range(3,25), ref=[2,3,4])

def cosspace(a, b, n=50):
    return (a + b)/2 + (b - a)/2 * (numpy.cos(numpy.linspace(0, numpy.pi, n)))

def fint_legendre_cos(f, a, b, n):
    x = cosspace(a, b, n)
    fx = f(x)
    P = vander_legendre(cosspace(-1,1,n))
    c = numpy.linalg.solve(P, fx)
    return c[0] * (b-a)

plot_accuracy(fint_legendre_cos, tests[:-1], range(3,25), ref=[2,3,4])

def fint_legendre(f, a, b, n):
    """Gauss-Legendre integration using Golub-Welsch algorithm"""
    beta = .5/numpy.sqrt(1-(2.*numpy.arange(1,n))**(-2))
    T = numpy.diag(beta,-1) + numpy.diag(beta, 1)
    D, V = numpy.linalg.eig(T)
    w = V[0,:]**2 * (b-a)
    x = (a+b)/2 + (b-a)/2 * D
    return w.dot(f(x))

fint_legendre(tests[0].f, -1, 1, 3)

plot_accuracy(fint_legendre, tests, range(3,25), ref=[2,3,4], plot=pyplot.semilogy)
pyplot.title('Gauss-Legendre Integration')

fint_legendre(lambda x: (x-.5)**9, -1, 1, n=5) - ((1-.5)**10/10 - (-1-.5)**10/10)



