get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean, sqrt
from scipy.stats import norm as ndist, binom

# stats60 specific
from code.probability import BoxModel, Binomial, RandomVariable, SumIntegerRV
from code import roulette
from code.week1 import standardize_right, standardize_left, normal_curve
from code.utils import probability_histogram
figsize = (8,8)

coin_trial = BoxModel(['H','T'])
coin_trial.mass_function

get_ipython().run_cell_magic('capture', '', "one_toss = plt.figure(figsize=figsize)\none_toss_ax = probability_histogram(Binomial(1, coin_trial, ['H']),\n                                    draw_bins=np.arange(2)-0.5,\n                                    alpha=0.5, facecolor='gray',\n                                    xlabel='Number of heads',\n                                    ylabel='% per head')[0]\none_toss_ax.set_xlim([-0.6,1.6])")

one_toss

get_ipython().run_cell_magic('capture', '', "two_toss = plt.figure(figsize=figsize)\ntwo_toss_ax = probability_histogram(Binomial(2, coin_trial, ['H']),\n                                    draw_bins=np.arange(3)-0.5,\n                                    alpha=0.5, facecolor='gray',\n                                    xlabel='Number of heads',\n                                    ylabel='% per head')[0]\ntwo_toss_ax.set_xlim([-0.6,2.6])\ntwo_toss_ax.legend()")

two_toss

two_draws = Binomial(2, BoxModel(['H','T']), ['H'])
two_draws.mass_function

two_draws.sample(5)

get_ipython().run_cell_magic('capture', '', "two_toss = plt.figure(figsize=figsize)\ntwo_toss_ax = probability_histogram(Binomial(2, coin_trial, ['H']),\n                                    bins=np.arange(4)-0.5,\n                                    alpha=0.5, facecolor='gray',\n                                    xlabel='Number of heads',\n                                    ylabel='% per head',\n                                    ndraws=500)[0]\ntwo_toss_ax.set_xlim([-0.6,2.6])\ntwo_toss_ax.legend()")

two_toss

get_ipython().run_cell_magic('capture', '', "tosses = {}\nntoss = 5\ntosses[ntoss] = plt.figure(figsize=figsize)\nprobability_histogram(Binomial(ntoss, coin_trial, ['H']),\n                                    bins=np.arange(7)-0.5,\n                                    alpha=0.5, facecolor='gray',\n                                    xlabel='Number of heads',\n                                    ylabel='% per heads',\n                                    ndraws=500)\ntosses[ntoss].gca().set_xlim([-0.6,5.6])\ntosses[ntoss].gca().legend()")

tosses[5]

get_ipython().run_cell_magic('capture', '', "ntoss = 30\ntosses[ntoss] = plt.figure(figsize=figsize)\nprobability_histogram(Binomial(ntoss, coin_trial, ['H']),\n                                    bins=np.arange(32)-0.5,\n                                    alpha=0.5, facecolor='gray',\n                                    xlabel='Number of heads',\n                                    ylabel='% per heads',\n                                    ndraws=500)\ntosses[ntoss].gca().set_xlim([5.6,25.6])\ntosses[ntoss].gca().legend()")

tosses[30]

roulette.examples['lucky numbers']

get_ipython().run_cell_magic('capture', '', "lucky_numbers = {}\nnbet = 10\nlucky_trial = roulette.examples['lucky numbers'].model\nlucky_numbers[nbet] = plt.figure(figsize=figsize)\nprobability_histogram(Binomial(nbet, lucky_trial, None),\n                                    bins=np.arange(12)-0.5,\n                                    alpha=0.5, facecolor='gray',\n                                    xlabel='Number of successes',\n                                    ylabel='% per success')\nlucky_numbers[nbet].gca().set_xlim([-0.6,5.6])\nlucky_numbers[nbet].gca().set_title('After %d bets' % nbet, fontsize=15)\nlucky_numbers[nbet].gca().legend()")

lucky_numbers[10]

get_ipython().run_cell_magic('capture', '', "nbet = 100\nlucky_numbers[nbet] = plt.figure(figsize=figsize)\nprobability_histogram(Binomial(nbet, lucky_trial, None),\n                                    bins=np.arange(12)-0.5,\n                                    alpha=0.5, facecolor='gray',\n                                    xlabel='Number of successes',\n                                    ylabel='% per success')\nlucky_numbers[nbet].gca().set_xlim([-0.6,20.6])\nlucky_numbers[nbet].gca().set_title('After %d bets' % nbet, fontsize=15)\nlucky_numbers[nbet].gca().legend()\n")

lucky_numbers[100]

get_ipython().run_cell_magic('capture', '', "nbet = 1000\nlucky_numbers[nbet] = plt.figure(figsize=figsize)\nprobability_histogram(Binomial(nbet, lucky_trial, None),\n                                    bins=np.arange(12)-0.5,\n                                    alpha=0.5, facecolor='gray',\n                                    xlabel='Number of successes',\n                                    ylabel='% per success')\nlucky_numbers[nbet].gca().set_xlim([50.6,110.6])\nlucky_numbers[nbet].gca().set_title('After %d bets' % nbet, fontsize=15)\nlucky_numbers[nbet].gca().legend()\n\n")

lucky_numbers[1000]

get_ipython().run_cell_magic('capture', '', "winnings = {}\nnbet = 10\nwinnings[nbet] = plt.figure(figsize=figsize)\n\nsuccesses = Binomial(nbet, roulette.examples['lucky numbers'])\ntotal = RandomVariable(successes, lambda wins : 100 + 110 * wins - 10 * (nbet - wins) )\nprobability_histogram(total, width=120, facecolor='gray',\n                      xlabel='Total (\\$)', ylabel='% per \\$')\nwinnings[nbet].gca().set_xlim([-60,610])\nwinnings[nbet].gca().set_title('After %d bets of 10\\$' % nbet, fontsize=15)")

winnings[10]

get_ipython().run_cell_magic('capture', '', "winnings = {}\nnbet = 100\nwinnings[nbet] = plt.figure(figsize=figsize)\n\nsuccesses = Binomial(nbet, roulette.examples['lucky numbers'])\ntotal = RandomVariable(successes, lambda wins : 100 + 110 * wins - 10 * (nbet - wins) )\nprobability_histogram(total, width=120, facecolor='gray',\n                      xlabel='Total (\\$)', ylabel='% per \\$')\n\nwinnings[nbet].gca().set_title('After %d bets of 10\\$' % nbet, fontsize=15)")

winnings[100]

get_ipython().run_cell_magic('capture', '', "winnings = {}\nnbet = 1000\nwinnings[nbet] = plt.figure(figsize=figsize)\n\nsuccesses = Binomial(nbet, roulette.examples['lucky numbers'])\ntotal = RandomVariable(successes, lambda wins : 100 + 110 * wins - 10 * (nbet - wins) )\nprobability_histogram(total, width=120, facecolor='gray',\n                      xlabel='Total (\\$)', ylabel='% per \\$')\n\nwinnings[nbet].gca().set_title('After %d bets of 10\\$' % nbet, fontsize=15)")

winnings[1000]

places = {}
for i in range(1,37) + ['0','00']:
    if i in [5]:
        places[i] = roulette.roulette_position(350,
                                               facecolor='green',
                                               bg_alpha=None,
                                               fontsize=90)
    else:
        places[i] = roulette.roulette_position(-10,
                                               facecolor='red',
                                               bg_alpha=None,
                                               fontsize=90)
winnings = roulette.roulette_table(places)
from IPython.core.display import HTML

HTML(winnings)

get_ipython().run_cell_magic('capture', '', 'with plt.xkcd():\n    winnings_stand = plt.figure(figsize=(10,5))\n    standardize_right(100, -52, 576, units="Total amount", standardized=True,\n                      data=False)')

winnings_stand

get_ipython().run_cell_magic('capture', '', "normal_fig_5 = plt.figure(figsize=figsize)\nax = normal_curve()\ninterval = np.linspace(0.26, 4, 101)\nax.fill_between(interval, 0*interval, ndist.pdf(interval),\n                hatch='+', color='green', alpha=0.5)\nax.set_title('The green area is %0.1f%%' % (100 * ndist.sf(0.27)), fontsize=20, color='green')")

normal_fig_5

get_ipython().run_cell_magic('capture', '', "winnings = {}\nnbet = 10\nwinnings[nbet] = plt.figure(figsize=figsize)\n\nsuccesses = Binomial(nbet, roulette.examples['lucky numbers'])\ntotal = RandomVariable(successes, lambda wins : 100 + 110 * wins - 10 * (nbet - wins) )\nax, avg, sd = probability_histogram(total, width=120, facecolor='gray',\n                      xlabel='Total (\\$)', ylabel='% per \\$')\n\nwinnings[nbet].gca().set_title('After %d bets of 10\\$' % nbet, fontsize=15)\nnormal_curve(mean=avg, SD=sd, ax=ax, alpha=0.3, facecolor='green', color='green',\n             xlabel=None, ylabel=None)")

winnings[10]

get_ipython().run_cell_magic('capture', '', "winnings = {}\nnbet = 100\nwinnings[nbet] = plt.figure(figsize=figsize)\n\nsuccesses = Binomial(nbet, roulette.examples['lucky numbers'])\ntotal = RandomVariable(successes, lambda wins : 100 + 110 * wins - 10 * (nbet - wins) )\nax, avg, sd = probability_histogram(total, width=120, facecolor='gray',\n                      xlabel='Total (\\$)', ylabel='% per \\$')\n\nwinnings[nbet].gca().set_title('After %d bets of 10\\$' % nbet, fontsize=15)\nnormal_curve(mean=avg, SD=sd, ax=ax, alpha=0.3, facecolor='green', color='green',\n             xlabel=None, ylabel=None)")

winnings[100]

get_ipython().run_cell_magic('capture', '', "winnings = {}\nnbet = 1000\nwinnings[nbet] = plt.figure(figsize=figsize)\n\nsuccesses = Binomial(nbet, roulette.examples['lucky numbers'])\ntotal = RandomVariable(successes, lambda wins : 100 + 110 * wins - 10 * (nbet - wins) )\nax, avg, sd = probability_histogram(total, width=120, facecolor='gray',\n                      xlabel='Total (\\$)', ylabel='% per \\$')\n\nwinnings[nbet].gca().set_title('After %d bets of 10\\$' % nbet, fontsize=15)\nnormal_curve(mean=avg, SD=sd, ax=ax, alpha=0.3, facecolor='green', color='green',\n             xlabel=None, ylabel=None)")

winnings[1000]

get_ipython().run_cell_magic('capture', '', "tosses = {}\nntoss = 100\ntosses[ntoss] = plt.figure(figsize=figsize)\nax, avg, sd = probability_histogram(Binomial(ntoss, coin_trial, ['H']),\n                                    bins=np.arange(7)-0.5,\n                                    alpha=0.5, facecolor='gray',\n                                    xlabel='Number of heads',\n                                    ylabel='% per head',\n                                    ndraws=500)\nnormal_curve(mean=avg, SD=sd, ax=ax, alpha=0.3, facecolor='green', color='green',\n             xlabel=None, ylabel=None)")

tosses[100]

interval = np.linspace(0, 44.5, 101)
ax.fill_between(interval, 0*interval, ndist.pdf((interval - avg) / sd) / sd,
                hatch='+', color='red', alpha=0.5)
correction = np.linspace(44,44.5,101)
ax.fill_between(correction, 0*correction, ndist.pdf((correction - avg) / sd) / sd,
                hatch='+', color='blue', alpha=0.8)
ax.set_title('Using continuity correction', fontsize=20, color='red')
ax.set_xlim([ax.get_xlim()[0],50])

tosses[100]

get_ipython().run_cell_magic('capture', '', 'with plt.xkcd():\n    heads_stand = plt.figure(figsize=figsize)\n    standardize_left(44.5, avg, sd, units="Heads", standardized=True,\n                     data=False)')

heads_stand

get_ipython().run_cell_magic('capture', '', "normal_fig_45 = plt.figure(figsize=figsize)\nax = normal_curve()\ninterval = np.linspace(-4,-1.10, 101)\nax.fill_between(interval, 0*interval, ndist.pdf(interval),\n                hatch='+', color='green', alpha=0.5)\nax.set_title('The green area is %0.1f%%' % (100 * ndist.cdf(-1.10)), fontsize=20, color='green')")

normal_fig_45

toss100 = Binomial(100, coin_trial, ['H'])
print("Exact: %s " % `(sum([toss100.mass_function[i] for i in range(45)]), binom.cdf(44,100,0.5))`)

get_ipython().run_cell_magic('capture', '', "normal_fig_without = plt.figure(figsize=figsize)\nax = normal_curve()\nZ = (45 - avg) / sd\nprint(Z)\ninterval = np.linspace(-4, Z, 101)\nax.fill_between(interval, 0*interval, ndist.pdf(interval),\n                hatch='+', color='green', alpha=0.5)\nax.set_title('The green area is %0.1f%%' % (100 * ndist.cdf(Z)), fontsize=20, color='green')")

normal_fig_without

get_ipython().run_cell_magic('capture', '', "normal_fig = plt.figure(figsize=figsize)\nax = normal_curve()\ninterval = np.linspace(-2.1,-1.9, 101)\nax.fill_between(interval, 0*interval, ndist.pdf(interval),\n                hatch='+', color='green', alpha=0.5)\nax.set_title('The green area is %0.1f%%' % (100 * (ndist.cdf(-1.9) - ndist.cdf(-2.10))), fontsize=20, color='green')\n")

normal_fig

Binomial(100, coin_trial, ['H']).mass_function[40]

get_ipython().run_cell_magic('capture', '', "normal_fig_80 = plt.figure(figsize=figsize)\nax = normal_curve()\ninterval = np.linspace(4.5 / np.sqrt(20), 4, 101)\nax.fill_between(interval, 0*interval, ndist.pdf(interval),\n                hatch='+', color='green', alpha=0.5)\nax.set_title('The green area is %0.1f%%' % (100 * ndist.sf(4.5 / np.sqrt(20))), fontsize=20, color='green')")

normal_fig_80

get_ipython().run_cell_magic('capture', '', "normal_fig = plt.figure(figsize=figsize)\nax = normal_curve()\ninterval = np.linspace(4.5 / np.sqrt(20), 5.5 / np.sqrt(20), 101)\nax.fill_between(interval, 0*interval, ndist.pdf(interval),\n                hatch='+', color='green', alpha=0.5)\nax.set_title('The green area is %0.1f%%' % (100 * (ndist.sf(4.5 / np.sqrt(20)) - ndist.sf(5.5 / np.sqrt(20)))), fontsize=20, color='green')")

normal_fig

Binomial(80, coin_trial, ['H']).mass_function[45], binom.pmf(45, 80, 0.5)

get_ipython().run_cell_magic('capture', '', "lopsided = {}\nndraw = 3 \nlopsided[ndraw] = plt.figure(figsize=figsize)\nmass = np.array([0,1,1,0,0,0,0,1])/3.\nrv = SumIntegerRV(mass, 3)\nax, avg, sd = probability_histogram(rv,\n                                    facecolor='gray')\nnormal_curve(mean=avg, SD=sd, ax=ax, alpha=0.3, facecolor='green', color='green',\n             xlabel=None, ylabel=None)\nax.set_ylim([0,1.1*max(rv.mass_function.values())])\nax.set_title('%d draws from [1,2,7]' % ndraw, fontsize=17)")

lopsided[3]

get_ipython().run_cell_magic('capture', '', "ndraw = 10\n\nlopsided[ndraw] = plt.figure(figsize=figsize)\nmass = np.array([0,1,1,0,0,0,0,1])/3.\nrv = SumIntegerRV(mass, ndraw)\nax, avg, sd = probability_histogram(rv,\n                                    facecolor='gray')\nnormal_curve(mean=avg, SD=sd, ax=ax, alpha=0.3, facecolor='green', color='green',\n             xlabel=None, ylabel=None)\nax.set_ylim([0,1.1*max(rv.mass_function.values())])\nax.set_title('%d draws from [1,2,7]' % ndraw, fontsize=17)")

lopsided[10]

get_ipython().run_cell_magic('capture', '', "ndraw = 30\n\nlopsided[ndraw] = plt.figure(figsize=figsize)\nmass = np.array([0,1,1,0,0,0,0,1])/3.\nrv = SumIntegerRV(mass, ndraw)\nax, avg, sd = probability_histogram(rv,\n                                    facecolor='gray')\nnormal_curve(mean=avg, SD=sd, ax=ax, alpha=0.3, facecolor='green', color='green',\n             xlabel=None, ylabel=None)\nax.set_ylim([0,1.1*max(rv.mass_function.values())])\nax.set_title('%d draws from [1,2,7]' % ndraw, fontsize=17)")

lopsided[30]

get_ipython().run_cell_magic('capture', '', "ndraw = 50\n\nlopsided[ndraw] = plt.figure(figsize=figsize)\nmass = np.array([0,1,1,0,0,0,0,1])/3.\nrv = SumIntegerRV(mass, ndraw)\nax, avg, sd = probability_histogram(rv,\n                                    facecolor='gray')\nnormal_curve(mean=avg, SD=sd, ax=ax, alpha=0.3, facecolor='green', color='green',\n             xlabel=None, ylabel=None)\nax.set_ylim([0,1.1*max(rv.mass_function.values())])\nax.set_title('%d draws from [1,2,7]' % ndraw, fontsize=17)")

lopsided[50]

