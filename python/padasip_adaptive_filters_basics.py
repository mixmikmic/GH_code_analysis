import numpy as np
import matplotlib.pylab as plt

import padasip as pa

get_ipython().magic('matplotlib inline')
plt.style.use('ggplot') # nicer plots
np.random.seed(52102) # always use the same random seed to make results comparable
get_ipython().magic('config InlineBackend.print_figure_kwargs = {}')

u = np.arange(0,10,1)
u

x = pa.input_from_history(u, 4)
x

n = 4
x = pa.input_from_history(u, n, bias=True)
x

N = len(u)
n = 4
N - n + 1

len(x)

# signals creation: u, v, d
N = 1000
n = 15
u = np.sin(np.arange(0, N/10., N/10000.))
v = np.random.random(N) - 0.5
d = u + v

# filtering
x = pa.input_from_history(d, n)[:-1]
d = d[n:]
u = u[n:]
y, e, w = pa.rls_filter(d, x, mu=0.95)

# error estimation
MSE_d = np.dot(u-d, u-d) / float(len(u))
MSE_y = np.dot(u-y, u-y) / float(len(u))

# results
plt.figure(figsize=(12.5,6))
plt.plot(u, "r:", linewidth=4, label="original")
plt.plot(d, "b", label="noisy, MSE: {}".format(MSE_d))
plt.plot(y, "g", label="filtered, MSE: {}".format(MSE_y))
plt.xlim(800,900)
plt.legend()
plt.tight_layout()
plt.show()

# creation of x and d
N = 200
x = np.random.random((N, 4))
v = np.random.random(N) - 0.5
d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3] + v

# identification
y, e, w = pa.nlms_filter(d, x, mu=0.5)

# show results
plt.figure(figsize=(12.5,9))
plt.subplot(211);plt.title("Adaptation");plt.xlabel("Number of iteration [-]")
plt.plot(d,"b", label="d - target")
plt.plot(y,"g", label="y - output");plt.legend()

plt.subplot(212); plt.title("Filter error"); plt.xlabel("Number of iteration [-]")
plt.plot(pa.misc.logSE(e),"r", label="Squared error [dB]");plt.legend()
plt.tight_layout()
plt.show()
print("And the resulting coefficients are: {}".format(w))

# creation of u, x and d
N = 100
u = np.random.random(N)
d = np.zeros(N)
for k in range(3, N):
    d[k] = 2*u[k] + 0.1*u[k-1] - 4*u[k-2] + 0.5*u[k-3]
d = d[3:]

# identification
x = pa.input_from_history(u, 4)
y, e, w = pa.rls_filter(d, x, mu=0.1)

# show results
plt.figure(figsize=(12.5,9))
plt.subplot(211);plt.title("Adaptation");plt.xlabel("Number of iteration [-]")
plt.plot(d,"b", label="d - target")
plt.plot(y,"g", label="y - output");plt.legend()
plt.subplot(212);plt.title("Filter error");plt.xlabel("Number of iteration [-]")
plt.plot(pa.misc.logSE(e),"r", label="Squared error [db]");plt.legend()
plt.tight_layout()
plt.show()

