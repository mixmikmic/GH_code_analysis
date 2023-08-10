get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext rpy2.ipython')
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean, sqrt, std, fabs
from scipy.stats import t as tdist

# stats60 specific
from scipy.stats import norm as ndist
from code import roulette
from code.week1 import normal_curve
from code.probability import ProbabilitySpace, BoxModel, Normal
from code.week7 import studentT_curve
figsize = (8,8)

get_ipython().run_cell_magic('capture', '', "normal_fig = plt.figure(figsize=figsize)\nax = normal_curve()\ninterval = np.linspace(-4,-2.2, 101)\nax.fill_between(interval, 0*interval, ndist.pdf(interval),\n                hatch='+', color='green', alpha=0.5)\nax.set_title('The green area is %0.1f%%' % (100 * ndist.cdf(-2.2)), fontsize=20, color='green')")

normal_fig

get_ipython().run_cell_magic('capture', '', 'class BloodPressure(ProbabilitySpace):\n\n    alpha = 0.1\n    ptsize = 5\n    sample_ptsize = 60\n\n    def __init__(self, draw_fig=True):\n        self.npop, self.ndraw = 5000, 50\n        self.box = BoxModel(np.random.randint(-15, 1,\n                                              size=(self.npop,)),\n                                              replace=True)\n        self.X = (np.mgrid[0:1:10j,0:1:5j].reshape((2,50)) +\n                  np.random.sample((2,50)) * 0.05)\n        self.BG = np.random.sample((self.npop,2))\n        self.X = self.X.T\n        self.draw_fig = draw_fig\n        if draw_fig:\n            self.draw()\n\n    def draw(self, color={\'R\':\'red\',\'B\':\'blue\'}):\n        self.figure.clf()\n        ax, X, BG = self.axes, self.X, self.BG\n        ax.scatter(BG[:,0], BG[:,1], s=self.ptsize, color=\'gray\', alpha=self.alpha)\n        ax.set_xticks([]);    ax.set_xlim([-0.1,1.1])\n        ax.set_yticks([]);    ax.set_ylim([-0.1,1.1])\n\n    @property\n    def figure(self):\n        if not hasattr(self, "_figure"):\n            self._figure = plt.figure(figsize=figsize)\n        self._axes = self._figure.gca()\n        return self._figure\n\n    @property\n    def axes(self):\n        self.figure\n        return self._axes\n    \n    def draw_sample_pts(self, bgcolor={\'R\':\'red\',\'B\':\'blue\'},\n                        color={\'R\':\'red\',\'B\':\'blue\'}):\n        self.draw(color=bgcolor)\n        ax, X, sample = self.axes, self.X, self._sample\n        mean, sd = self.outcome\n        for i in range(50):\n            ax.text(X[i,0], X[i,1], \'%d\' % sample[i], color=\'red\')\n        ax.set_title("average(sample)=%0.1f, SD(sample)=%0.1f" % (np.mean(sample), np.std(sample)), fontsize=15)\n        return self.figure\n\n    def trial(self, bgcolor={\'R\':\'red\',\'B\':\'blue\'},\n              color={\'R\':\'red\',\'B\':\'blue\'}):\n        self._sample = self.box.sample(self.ndraw)\n        self.outcome = np.mean(self._sample), np.std(self._sample)\n        if self.draw_fig:\n            self.draw_sample_pts(color=color, bgcolor=bgcolor)\n        return self.outcome\n\nnp.random.seed(1)    \nBP = BloodPressure()\nBP.trial()')

BP.figure

4.7 / sqrt(50)

get_ipython().run_cell_magic('capture', '', "normal_fig2 = plt.figure(figsize=figsize)\nax = normal_curve()\ninterval = np.linspace(-4,-.8, 101)\nax.fill_between(interval, 0*interval, ndist.pdf(interval),\n                hatch='+', color='green', alpha=0.5)\ninterval = np.linspace(.8,4, 101)\nax.fill_between(interval, 0*interval, ndist.pdf(interval),\n                hatch='+', color='green', alpha=0.5)\n\nax.set_title('The green area is %0.1f%%' % (2 * 100 * ndist.sf(.8)), fontsize=20, color='green')\n")

normal_fig2

get_ipython().run_cell_magic('capture', '', "normal_fig3 = plt.figure(figsize=figsize)\nax = normal_curve()\ninterval = np.linspace(-4,-1.65, 101)\nax.fill_between(interval, 0*interval, ndist.pdf(interval),\n                hatch='+', color='green', alpha=0.5)\nax.set_title('The green area is %0.0f%%' % (100 * ndist.cdf(-1.65)), fontsize=20, color='green')")

normal_fig3

get_ipython().run_cell_magic('capture', '', "normal_fig4 = plt.figure(figsize=figsize)\nax = normal_curve()\ninterval = np.linspace(-4,-2, 101)\nax.fill_between(interval, 0*interval, ndist.pdf(interval),\n                hatch='+', color='green', alpha=0.5)\ninterval = np.linspace(2,4, 101)\nax.fill_between(interval, 0*interval, ndist.pdf(interval),\n                hatch='+', color='green', alpha=0.5)\n\nax.set_title('The green area is %0.0f%%' % (2 * 100 * ndist.cdf(-2)), fontsize=20, color='green')")

normal_fig4

get_ipython().run_cell_magic('capture', '', "normal_fig5 = plt.figure(figsize=figsize)\nax = normal_curve()\ninterval = np.linspace(1.65,4, 101)\nax.fill_between(interval, 0*interval, ndist.pdf(interval),\n                hatch='+', color='green', alpha=0.5)\nax.set_title('The green area is %0.0f%%' % (100 * ndist.sf(1.65)), fontsize=20, color='green')")

normal_fig5

normal_fig3

normal_fig5

normal_fig4

def Rplus(thetahat, theta0, SE_thetahat):
    return np.greater(thetahat, theta0 + 1.65 * SE_thetahat)
theta_generator = Normal(50, 1.25)
thetahat = theta_generator.trial()
thetahat, Rplus(thetahat, 50, 1.25)

Tsample = theta_generator.sample(10000)
mean(Rplus(Tsample, 50, 1.25))

theta_generator2 = Normal(48, 1.25)
Tsample2 = theta_generator2.sample(10000)
mean(Rplus(Tsample2, 50, 1.25))

def Rminus(thetahat, theta0, SE_thetahat):
    return np.less(thetahat, theta0 - 1.65 * SE_thetahat)
mean(Rminus(Tsample, 50, 1.25))

def snooping_test(thetahat, theta0, SD_thetahat):
    test_pos = np.greater(thetahat, theta0) * Rplus(thetahat, theta0, SD_thetahat)
    test_neg = np.less(thetahat, theta0) * Rminus(thetahat, theta0, SD_thetahat)
    return test_pos + test_neg

mean(snooping_test(Tsample, 50, 1.25))

get_ipython().run_cell_magic('capture', '', "df=4\nnormal_fig6 = plt.figure(figsize=figsize)\nax = normal_fig6.gca()\nnormal_curve(ax=ax, label='Normal', color='blue', alpha=0.)\nstudentT_curve(ax=ax, label='$T_{%d}$' % df, color='green', alpha=0., df=df)\nax.set_title('Comparison of normal curve to $T_{%d}$' % df, fontsize=15)\nax.legend()")

normal_fig6

get_ipython().run_cell_magic('capture', '', "df = 4 \nnormal_fig7 = plt.figure(figsize=figsize)\nax = normal_curve(alpha=0., color='blue')\ninterval = np.linspace(-4,ndist.ppf(0.025), 101)\nax.fill_between(interval, 0*interval, ndist.pdf(interval),\n                hatch='+', color='blue', alpha=0.5)\ninterval = np.linspace(ndist.ppf(0.975),4, 101)\nax.fill_between(interval, 0*interval, ndist.pdf(interval),\n                hatch='+', color='blue', alpha=0.5)\n\nstudentT_curve(ax=ax, alpha=0., color='green', df=df)\ninterval = np.linspace(-4,tdist.ppf(0.025, df), 101)\nax.fill_between(interval, 0*interval, tdist.pdf(interval, df),\n                hatch='+', color='green', alpha=0.2)\ninterval = np.linspace(tdist.ppf(0.975, df),4, 101)\nax.fill_between(interval, 0*interval, tdist.pdf(interval, df),\n                hatch='+', color='green', alpha=0.2)\n")

normal_fig7

get_ipython().run_cell_magic('capture', '', "df=20\nnormal_fig8 = plt.figure(figsize=figsize)\nax = normal_fig8.gca()\nnormal_curve(ax=ax, label='Normal', color='blue', alpha=0.)\nstudentT_curve(ax=ax, label='$T_{%d}$' % df, color='green', alpha=0., df=df)\nax.set_title('Comparison of normal curve to $T_{%d}$' % df, fontsize=15)\nax.legend()\n")

normal_fig8

get_ipython().run_cell_magic('capture', '', "df = 20\nnormal_fig9 = plt.figure(figsize=figsize)\nax = normal_curve(alpha=0., color='blue')\ninterval = np.linspace(-4,ndist.ppf(0.025), 101)\nax.fill_between(interval, 0*interval, ndist.pdf(interval),\n                hatch='+', color='blue', alpha=0.5)\ninterval = np.linspace(ndist.ppf(0.975),4, 101)\nax.fill_between(interval, 0*interval, ndist.pdf(interval),\n                hatch='+', color='blue', alpha=0.5)\n\nstudentT_curve(ax=ax, alpha=0., color='green', df=df)\ninterval = np.linspace(-4,tdist.ppf(0.025, df), 101)\nax.fill_between(interval, 0*interval, tdist.pdf(interval, df),\n                hatch='+', color='green', alpha=0.2)\ninterval = np.linspace(tdist.ppf(0.975, df),4, 101)\nax.fill_between(interval, 0*interval, tdist.pdf(interval, df),\n                hatch='+', color='green', alpha=0.2)\n")

normal_fig9

