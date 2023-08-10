get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

# Use seaborn styles for nice-looking plots
import seaborn; seaborn.set()

x = np.linspace(-1, 1, 1000)
for beta in [0.5, 1, 5]:
    plt.plot(x, 0.5 * (1 + np.tanh(beta * x)),
             label=r'$\beta = {0:.1f}$'.format(beta))
plt.legend(loc='upper left', fontsize=18)
plt.xlabel('$x$', size=18); plt.ylabel(r'$\phi(x;\beta)$', size=18);

class HipsterStep(object):
    """Class to implement hipster evolution
    
    Parameters
    ----------
    initial_style : length-N array
        values > 0 indicate one style, while values <= 0 indicate the other.
    is_hipster : length-N array
        True or False, indicating whether each person is a hipster
    influence_matrix : N x N array
        Array of non-negative values. influence_matrix[i, j] indicates
        how much influence person j has on person i
    delay_matrix : N x N array
        Array of positive integers. delay_matrix[i, j] indicates the
        number of days delay between person j's influence on person i.
    """
    def __init__(self, initial_style, is_hipster,
                 influence_matrix, delay_matrix,
                 beta=1, rseed=None):
        self.initial_style = initial_style
        self.is_hipster = is_hipster
        self.influence_matrix = influence_matrix
        self.delay_matrix = delay_matrix
        
        self.rng = np.random.RandomState(rseed)
        self.beta = beta
        
        # make s array consisting of -1 and 1
        self.s = -1 + 2 * (np.atleast_2d(initial_style) > 0)
        N = self.s.shape[1]
        
        # make eps array consisting of -1 and 1
        self.eps = -1 + 2 * (np.asarray(is_hipster) > 0)
        
        # create influence_matrix and delay_matrix
        self.J = np.asarray(influence_matrix, dtype=float)
        self.tau = np.asarray(delay_matrix, dtype=int)
        
        # validate all the inputs
        assert self.s.ndim == 2
        assert self.s.shape[1] == N
        assert self.eps.shape == (N,)
        assert self.J.shape == (N, N)
        assert np.all(self.J >= 0)
        assert np.all(self.tau > 0)

    @staticmethod
    def phi(x, beta):
        return 0.5 * (1 + np.tanh(beta * x))
            
    def step_once(self):
        N = self.s.shape[1]
        
        # iref[i, j] gives the index for the j^th individual's
        # time-delayed influence on the i^th individual
        iref = np.maximum(0, self.s.shape[0] - self.tau)
        
        # sref[i, j] gives the previous state of the j^th individual
        # which affects the current state of the i^th individual
        sref = self.s[iref, np.arange(N)]

        # m[i] is the mean of weighted influences of other individuals
        m = (self.J * sref).sum(1) / self.J.sum(1)
        
        # From m, we use the sigmoid function to compute a transition probability
        transition_prob = self.phi(-self.eps * m * self.s[-1], beta=self.beta)
        
        # Now choose steps stochastically based on this probability
        new_s = np.where(transition_prob > self.rng.rand(N), -1, 1) * self.s[-1]
        
        # Add this to the results, and return
        self.s = np.vstack([self.s, new_s])
        return self.s
    
    def step(self, N):
        for i in range(N):
            self.step_once()
        return self.s
    
    def trend(self, hipster=True, return_std=True):
        if hipster:
            subgroup = self.s[:, self.eps > 0]
        else:
            subgroup = self.s[:, self.eps < 0]
            
        return subgroup.mean(1), subgroup.std(1)

def plot_results(Npeople=500, Nsteps=200,
                 hipster_frac=0.8, initial_state_frac=0.5,
                 delay=20, log10_beta=0.5, rseed=42):
    rng = np.random.RandomState(rseed)
    
    initial_state = (rng.rand(1, Npeople) > initial_state_frac)
    is_hipster = (rng.rand(Npeople) > hipster_frac)

    influence_matrix = abs(rng.randn(Npeople, Npeople))
    influence_matrix.flat[::Npeople + 1] = 0
    
    delay_matrix = 1 + rng.poisson(delay, size=(Npeople, Npeople))

    h = HipsterStep(initial_state, is_hipster,
                    influence_matrix, delay_matrix=delay_matrix,
                    beta=10 ** log10_beta, rseed=rseed)
    h.step(Nsteps)
    
    def beard_formatter(y, loc):
        if y == 1:
            return 'bushy-\nbeard'
        elif y == -1:
            return 'clean-\nshaven'
        else:
            return ''
        
    t = np.arange(Nsteps + 1)

    fig, ax = plt.subplots(2, sharex=True, figsize=(8, 6))
    ax[0].imshow(h.s.T, cmap='binary', interpolation='nearest')
    ax[0].set_ylabel('individual')
    ax[0].axis('tight')
    ax[0].grid(False)
    
    mu, std = h.trend(True)
    ax[1].plot(t, mu, c='red', label='hipster')
    ax[1].fill_between(t, mu - std, mu + std, color='red', alpha=0.2)
    
    mu, std = h.trend(False)
    ax[1].plot(t, mu, c='blue', label='conformist')
    ax[1].fill_between(t, mu - std, mu + std, color='blue', alpha=0.2)
    
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('Trend')
    ax[1].legend(loc='best')
    ax[1].set_ylim(-1.1, 1.1);
    ax[1].set_xlim(0, Nsteps)
    ax[1].yaxis.set_major_formatter(plt.FuncFormatter(beard_formatter))

plot_results(hipster_frac=0.1)

plot_results(hipster_frac=0.5)

plot_results(hipster_frac=0.8)

from IPython.html.widgets import interact, fixed

interact(plot_results, hipster_frac=(0.0, 1.0), delay=(1, 50),
         initial_state_frac=(0.0, 1.0), log10_beta=(-2.0, 2.0),
         Npeople=fixed(500), Nsteps=fixed(200), rseed=fixed(42));

