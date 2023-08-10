get_ipython().magic('pylab inline')
plt.rcParams['figure.figsize'] = (14, 10)

from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

k, H, n, tc, q = 2.5, 1e-6, 100, 35000, -0.02

x = linspace(0, tc, n)  # pozice uzlu
dx = tc / (n - 1)       # vzdalenost uzlu
d = ones(n)

A = spdiags([d, -2*d, d], [-1, 0, 1], n, n, 'csr')
b = -d * H * dx**2 / k

A[0, :2] = [1, 0]
b[0] = 0
A[-1, -2:] = [2, -2]
b[-1] += 2 * q * dx / k

ti = spsolve(A, b)

plot(ti, x)
ylim(tc, 0);

ti[logical_and(x<10000, x>5000)] = 700

plot(ti, x)
ylim(tc, 0);

k, H, n, tc, q, rho, c = 2.5, 1e-6, 100, 35000, -0.02, 2800, 900

x = linspace(0, tc, n)  # pozice uzlu
dx = tc / (n - 1)       # vzdalenost uzlu
d = ones(n)
kappa = k / (rho * c)

dt = 0.9*dx**2/(2 * kappa)
u = kappa*dt/dx**2

A = spdiags([d*u, d*(1 - 2*u), d*u], [-1, 0, 1], n, n, 'csr')
b = d * H * dt / (rho *c)

A[0, :2] = [1, 0]
b[0] = 0
A[-1, -2] = 2*u
b[-1] = dt*(H*dx - 2*q)/(rho * c *dx)

t = ti
res = []
tt = []
total_time = 0
plot(ti ,x)
for j in range(10):
    for i in range(100):
        t = A.dot(t) + b
        res.append(t)
        total_time += dt
        tt.append(total_time/(365.25*24*3600))
    plot(t, x)
ylim(tc, 0);
res = asarray(res)

total_time/(365.25*24*3600)

plot(tt, res[:,31])
xlabel('Time [years]')
ylabel('Temperature')



