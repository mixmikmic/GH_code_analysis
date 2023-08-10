import numpy as np
import scipy.stats as st
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
mpl.rcParams['figure.dpi'] = 300
import seaborn as sns
sns.set_style('white')
sns.set_context('talk')

num_mc = 10000
s = 0.
s2 = 0.
for i in range(num_mc):
    x = np.random.randn()
    if x > 4.5:
        s += 1.
        s2 += 1.
I_n = s / num_mc
V_n = (s2 / num_mc - I_n ** 2) / num_mc
print('I_n = %1.2e +- %1.2e' % (I_n, 2 * np.sqrt(V_n)))

fig, ax = plt.subplots()
x = np.linspace(-4., 10., 200)
p = st.norm.pdf(x)
q = np.exp(-(x-4.5))
q[x < 4.5] = 0.
ax.plot(x, p, label='$p(x)$')
ax.plot(x, q, label='$q(x)$')
ax.set_xlabel('$x$')
ax.set_ylabel('Probability density')
plt.legend(loc='best');

num_mc = 10000
s = 0.
s2 = 0.
for i in range(num_mc):
    x = np.random.exponential() + 4.5 # Just add 4.5 to a standard exponential
    f = np.exp(-0.5 * x ** 2 + x - 4.5) / np.sqrt(2. * np.pi)
    s += f
    s2 += f ** 2
I_n = s / num_mc
V_n = (s2 / num_mc - I_n ** 2) / num_mc
print('I_n = %1.4e +- %1.4e' % (I_n, 2*np.sqrt(V_n)))

# How many times to try
num_tries = 10
# How many MC steps to take per try 
num_mc = 1000
# For plotting purposes
fig, ax = plt.subplots()
for i in range(num_tries):
    I = np.ndarray((num_mc, ))
    s = 0.
    for j in range(num_mc):
        x = np.random.randn()
        if x >= 0.:
            y = x * np.exp(-x) / st.norm.pdf(x)
            s += y
        I[j] = s / (j + 1)
    plt.plot(np.arange(1, num_mc + 1), I, color=sns.color_palette()[0], lw=1)
ax.set_xlabel('$n$')
ax.set_ylabel('$I_n$')
plt.plot(np.arange(1, num_mc + 1), np.ones((num_mc, )), color='r');

import design # Library the implements latin hyper-cube designs you must install it: pip install py-design
fig, ax = plt.subplots()
ax.grid(which='major')
for i in range(1):
    X = design.latin_center(5, 2)
    ax.scatter(X[:,0], X[:,1], 50., color=sns.color_palette()[i])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$');

num_samples = 5
fig, ax = plt.subplots()
X_lhs = design.latin_center(num_samples, 2)
ax.scatter(X_lhs[:,0], X_lhs[:,1], 50., color=sns.color_palette()[0])
X_lhs = design.latin_center(num_samples, 2)
ax.grid(which='major')
ax.scatter(X_lhs[:,0], X_lhs[:,1], 50., color=sns.color_palette()[2])
ax.set_title('Latin hypercube samples')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$');
fig2, ax2 = plt.subplots()
X_unif = np.random.rand(num_samples, 2)
ax2.grid(which='major')
ax2.scatter(X_unif[:,0], X_unif[:,1], 50., color=sns.color_palette()[1])
X_unif = np.random.rand(num_samples, 2)
ax2.scatter(X_unif[:,0], X_unif[:,1], 50., color=sns.color_palette()[3])
ax2.set_title('Uniform samples')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$');

import scipy.integrate

class Ex1Solver(object):
    """
    An object that can solver the afforementioned ODE problem.
    It will work just like a multivariate function.
    """
    
    def __init__(self, nt=100, T=5):
        """
        This is the initializer of the class.
        
        Arguments:
            nt - The number of timesteps.
            T  - The final time.
        """
        self.nt = nt
        self.T = T
        self.t = np.linspace(0, T, nt) # The timesteps on which we will get the solution
        # The following are not essential, but they are convenient
        self.num_input = 2             # The number of inputs the class accepts
        self.num_output = nt           # The number of outputs the class returns
    
    def __call__(self, x):
        """
        This special class method emulates a function call.
        
        Arguments:
            x - A 1D numpy array with 2 elements. This represents the stochastic input x = (x1, x2).
        """
        def rhs(y, t, x1):
            """
            This is the right hand side of the ODE.
            """
            return -.1 * x1 * y
        # The initial condition
        y0 = [8 + 2 * x[1]]
        # We are ready to solve the ODE
        y = scipy.integrate.odeint(rhs, y0, self.t, args=(x[0],)).flatten()
        # The only strange thing here is the use of ``args`` to pass they x1 argument to the rhs().
        # That's it
        return y

# Here is how you can initialize it:
solver = Ex1Solver()
# You can access its arguments:
print "Number of timesteps:", solver.nt
print "Final time:", solver.T
print "Timesteps:", solver.t
print "Num inputs:", solver.num_input
print "Num outputs:", solver.num_output

# If you wish, you may intiialize it with a different initial arguments:
solver = Ex1Solver(nt=200, T=50)
print "Number of timesteps:", solver.nt
print "Final time:", solver.T
print "Timesteps:", solver.t
print "Num inputs:", solver.num_input
print "Num outputs:", solver.num_output

# Now let's evaluate the solver at a specific input.
# You can just use it as a function
x = [0.5, 0.5]
y = solver(x)
print y

# Let's plot it:
fig, ax = plt.subplots()
ax.plot(solver.t, y)
ax.set_xlabel('$t$')
ax.set_ylabel('$y(t)$')

# Now, let's just plot a few random samples.
fig, ax = plt.subplots()
for i in xrange(10):
    x = np.random.rand(2)
    y = solver(x)
    plt.plot(solver.t, y)
ax.set_xlabel('$t$')
ax.set_ylabel('$y(t)$');

# Let's now do Monte Carlo to compute the mean and the variance
# This is to accumulate the sum of all outputs
y_mc = np.zeros(solver.num_output)
# This is to accumlate the square of all outputs
y2_mc = np.zeros(solver.num_output)
# Pick the number of samples you wish to do:
num_samples = 10000
# Let's do it
data_mc = []
for i in xrange(num_samples):
    if i % 1000 == 0:
        print 'sample', i + 1, 'from', num_samples
    x = np.random.rand(2)
    y = solver(x)
    y_mc += y
    y2_mc += y ** 2
    data_mc.append(y)
data_mc = np.array(data_mc)
# Now we are ready for the mean estimate:
y_m_mc = y_mc / num_samples
# And the variance estimate
y_v_mc = y2_mc / num_samples - y_m_mc ** 2

# Let's plot the mean
fig, ax = plt.subplots()
ax.plot(solver.t, y_m_mc)
ax.set_xlabel('$t$')
ax.set_ylabel('$\mathbb{E}[y(t)]$')

# Let's plot the variance
fig, ax = plt.subplots()
ax.plot(solver.t, y_v_mc)
ax.set_xlabel('$t$')
ax.set_ylabel('$\mathbb{V}[y(t)]$')

# Now, let's draw the predictive envelop
# We need the standard deviation:
y_s_mc = np.sqrt(y_v_mc)
# A lower bound for the prediction
y_l_mc = np.percentile(data_mc, 2.75, axis=0)
# An upper bound for the prediction
y_u_mc = np.percentile(data_mc, 97.5, axis=0)
# And let's plot it:
fig, ax = plt.subplots()
ax.plot(solver.t, y_m_mc)
ax.fill_between(solver.t, y_l_mc, y_u_mc, alpha=0.25)

# Let's now do LHS to compute the mean and the variance
# This is to accumulate the sum of all outputs
y_lhs = np.zeros(solver.num_output)
# This is to accumlate the square of all outputs
y2_lhs = np.zeros(solver.num_output)
# Pick the number of samples you wish to do:
num_samples = 10000
# You have to create the LHS design prior to looping:
X = design.latin_center(num_samples, 2)
# Let's do it
data_lhs = []
for i in xrange(num_samples):
    if i % 1000 == 0:
        print 'sample', i + 1, 'from', num_samples
    x = X[i, :]
    y = solver(x)
    y_lhs += y
    y2_lhs += y ** 2
    data_lhs.append(y)
data_lhs = np.array(data_lhs)
# Now we are ready for the mean estimate:
y_m_lhs = y_lhs / num_samples
# And the variance estimate
y_v_lhs = y2_lhs / num_samples - y_m_lhs ** 2

# Let's plot the mean
plt.plot(solver.t, y_m_lhs)
plt.xlabel('$t$')
plt.ylabel('$\mathbb{E}[y(t)]$')

# Let's plot the variance
plt.plot(solver.t, y_v_lhs)
plt.xlabel('$t$')
plt.ylabel('$\mathbb{V}[y(t)]$')

# Now, let's draw the predictive envelop
# We need the standard deviation:
y_s_lhs = np.sqrt(y_v_lhs)
y_l_lhs = np.percentile(data_lhs, 2.75, axis=0)
# An upper bound for the prediction
y_u_lhs = np.percentile(data_lhs, 97.5, axis=0)
# And let's plot it:
plt.plot(solver.t, y_m_lhs)
plt.fill_between(solver.t, y_l_lhs, y_u_lhs, alpha=0.25)

def E_y(t):
    if t == 0:
        return 9.
    return (90*(1-np.exp(-0.1*t))) / t
E_y =  np.vectorize(E_y)

def V_y(t):
    if t == 0:
        return (2440*0.2)/6. - E_y(t) **2
    
    return (2440.*(1-np.exp(-0.2*t)))/(6.*t) - E_y(t)**2
V_y = np.vectorize(V_y)

# Let's start by doing some plots that compare the variance obtained by MC and LHS to the true one
plt.plot(solver.t, V_y(solver.t), label='True variance')
plt.plot(solver.t, y_v_mc, '--', label='MC (10,000)')
plt.plot(solver.t, y_v_lhs, '-.', label='LHS (10,000))')
plt.legend(loc='best')

def get_MC_rse(max_num_samples=100):
    """
    Get the maximum error of MC.
    """
    y_v_true = V_y(solver.t)
    y = np.zeros(solver.num_output)
    y2 = np.zeros(solver.num_output)
    n = []
    rse = []
    for i in xrange(max_num_samples):
        x = np.random.rand(2)
        y_sample = solver(x)
        y += y_sample
        y2 += y_sample ** 2
        if i % 1 == 0:    # Produce estimate every 100 steps
            n.append(i + 1)
            y_m = y / (i + 1)
            y_v = y2 / (i + 1) - y_m ** 2
            rse.append(np.linalg.norm(y_v_true - y_v))
    return n, rse

def _get_LHS_rse(max_num_samples=100):
    y_v_true = V_y(solver.t)
    y = np.zeros(solver.num_output)
    y2 = np.zeros(solver.num_output)
    n = []
    rse = []
    X = design.latin_center(max_num_samples, 2)
    for i in xrange(max_num_samples):
        x = X[i, :]
        y_sample = solver(x)
        y += y_sample
        y2 += y_sample ** 2
        if i % 1 == 0:    # Produce estimate every 100 steps
            n.append(i + 1)
            y_m = y / (i + 1)
            y_v = y2 / (i + 1) - y_m ** 2
            rse.append(np.linalg.norm(y_v_true - y_v))
    return n, rse

def get_LHS_rse(max_num_samples=100):
    n = []
    rse = []
    for i in xrange(max_num_samples):
        if i % 1 == 0:
            _n, _rse = _get_LHS_rse(i + 1)
            n.append(_n[-1])
            rse.append(_rse[-1])
    return n, rse

n_mc, rse_mc = get_MC_rse(50)
n_lhs, rse_lhs = get_LHS_rse(50)
plt.plot(n_mc, rse_mc, label='MC RSE')
plt.plot(n_lhs, rse_lhs, label='LHS RSE')
plt.xlabel('$n$')
plt.ylabel('RSE')
plt.legend(loc='best')

num_exper = 1
num_samples = 20
rse_mc_samples = []
rse_lhs_samples = []
for i in xrange(num_exper):
    print i
    n_mc, rse_mc = get_MC_rse(num_samples)
    n_lhs, rse_lhs = get_LHS_rse(num_samples)
    rse_mc_samples.append(rse_mc)
    rse_lhs_samples.append(rse_lhs)
rse_mc_samples = np.array(rse_mc_samples)
rse_lhs_samples = np.array(rse_lhs_samples)
rse_mc_m = np.mean(rse_mc_samples, axis=0)
rse_mc_l = np.percentile(rse_mc_samples, 2.75, axis=0)
rse_mc_u = np.percentile(rse_mc_samples, 97.5, axis=0)
rse_lhs_m = np.mean(rse_lhs_samples, axis=0)
rse_lhs_l = np.percentile(rse_lhs_samples, 2.75, axis=0)
rse_lhs_u = np.percentile(rse_lhs_samples, 97.5, axis=0)
plt.plot(n_mc, rse_mc_m, label='MC')
plt.plot(n_lhs, rse_lhs_m, '--', label='LHS')
plt.fill_between(n_mc, rse_mc_l, rse_mc_u, alpha=0.25)
plt.fill_between(n_lhs, rse_lhs_l, rse_lhs_u, color='g', alpha=0.25)
plt.legend(loc='best')
plt.xlabel('$n$')
plt.ylabel('RSE')

