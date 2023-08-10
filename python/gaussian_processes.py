import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import utls 

utls.reset_plots()
get_ipython().run_line_magic('matplotlib', 'inline')

def cov_matrix_function(x1,x2,l):
    """Use a squared exponential covariance with a fixed length scale
    :param x1: A double, parameter of the covariance function
    :param x2: A double, parameter of the covariance function
    :param l: A double, hyperparameter of the GP determining the length scale over which the correlation between neighbouring points decays
    Returns: Squared exponential covariance function
    """
    return np.exp(-(x1-x2)*(x1-x2)/l)

D = 90 # number of points along x where we will evaluate the GP. D = dimension of the cov matrix
x = np.linspace(-5,5,D)
ndraws = 5 # number of functions to draw from GP

cmap = plt.cm.jet

def sample_from_gp(l):
    """
    Sample from a Gaussian Process
    :param l: The length scale of the squared exponential GP
    Returns: A numpy array of length (D) as a draw from the GP
    """
    sigma = np.zeros((D,D))
    for i in range(D):
        for j in range(D):
            sigma[i,j] = cov_matrix_function(x[i],x[j],l)
    return sigma, np.random.multivariate_normal(np.zeros(D),sigma) # sample from the GP

def add_GP_draws_to_plot(ax, l):
    """Add a number of samples from a Gaussian process to a plot
    :param ax: A AxesSubplot object, the axes to plot on
    :param l: The length scale of the squared exponential GP
    """
    for k in range(ndraws):
        sigma, y = sample_from_gp(l)
        col = cmap(int(round((k+1)/float(ndraws)*(cmap.N))))
        ax.plot(x,y,'-',alpha=0.5,color=col, linewidth = 2)
    ax.set_xlabel('Input, $x$')
    ax.set_ylabel('Output, $f(x)$')
    ax.set_title('$l={}$'.format(l),fontsize=20)

fig, axs = plt.subplots(1,3,figsize=(3*5,5))
axs = axs.ravel()
add_GP_draws_to_plot(axs[0],0.1)
add_GP_draws_to_plot(axs[1],1)
add_GP_draws_to_plot(axs[2],10)
plt.tight_layout()

l = 1 
var_noise = 0.01

sigma_true, y_true = sample_from_gp(l) # The true function, a sample from a GP 

data_n = 10
data_indicies = np.random.choice(np.arange(int(round(0.1*D)),int(round(0.9*D))),data_n,replace=False)

data_y = y_true[data_indicies] + np.random.normal(loc=0.0,scale=np.sqrt(var_noise),size=data_n)
data_x = x[data_indicies]

K = np.zeros((data_n,data_n)) # make a covariance matrix
for i in range(data_n):
	for j in range(data_n):
		K[i,j] = cov_matrix_function(data_x[i],data_x[j],l) # squared exponential GP

means = np.zeros(D)
variances = np.zeros(D)
for i, xs in enumerate(x):
	k = cov_matrix_function(xs, data_x, l) 
	K_inv_n = np.linalg.inv( K + var_noise*np.identity(data_n) ) 
	v = np.dot(K_inv_n, data_y)
	mean = np.dot(k, v)

	v2 = np.dot(K_inv_n, k)
	var = cov_matrix_function(xs, xs, l) + var_noise - np.dot(k, v2)

	means[i] = mean
	variances[i] = var

p2 = plt.Rectangle((0, 0), 0.1, 0.1, fc="red", alpha = 0.3, ec = 'red')
p3 = mlines.Line2D([], [], color='red')

# Plot a 95% BCI using the 2 sigma rule for Normal distributions
fig, ax = plt.subplots()
ax.fill_between(x, means+2*np.sqrt(variances), means-2*np.sqrt(variances), color='red', alpha=0.3)
p1=ax.plot(data_x, data_y, 'kx')



ax.plot(x, y_true,'-r')
ax.set_xlabel('input, x')
ax.set_ylabel('output, y')
ax.legend([p1[0],p2, p3], ['Data', 'Posterior predictive distribution', 'True function'], prop={'size':8});

