get_ipython().magic('matplotlib qt4')
from __future__ import division

from collections import defaultdict

from models import tools, optimize, models, filters
from models.tests import PerformanceTest

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

sns.set_style("ticks", {"legend.frameon": True})
mpl.rcParams['axes.color_cycle'] = ['#02A5F4', 'orange', 'green']

data = tools.load_data(limit=40000, offset=2400000)

grad = optimize.NaiveDescent(data)

descent1 = grad.search_pfae(1.5, -2, step_size=3, maxiter=100, precision=0.005)

descent2 = grad.search_pfae(5, 0.5, step_size=2.5, maxiter=100, precision=0.005)

descent3 = grad.search_pfag(1.5, -2, step_size=20, maxiter=36, precision=0.005)

elo = models.EloModel()
pfae = models.PFAExt(elo, gamma=2.99622612646, delta=-0.476090204636)
pfae_test = PerformanceTest(pfae, data)
pfae_test.run()

pfae_test.results['train']

plt.figure(num=None, figsize=(5, 4.3), dpi=160)

def annotate(descent, number, mark, color, xadd, yadd):
    row = descent.params.loc[number]
    grad = descent.grads.loc[number]
    plt.annotate(r'$\gamma={}$, $\delta={}$'.format(round(row.gamma, 2), round(row.delta, 2)),
                 xy=(number, grad), xycoords='data',
                 xytext=(number + xadd, grad + yadd), textcoords='data',
                 bbox=dict(boxstyle="round", fc="w", linewidth=1, edgecolor=color))
    plt.plot(number, grad, mark, color=color, markeredgewidth=0, markersize=10)
    
#annotate(descent1, 1, 'go', 0.8, -0.006)
#annotate(descent1, 10, 'go', 0.8, -0.006)
annotate(descent1, 34, 'o', '#02A5F4', -15, -0.015)

#annotate(descent3, 1, 'ro', 0.7, 0.004)
#annotate(descent3, 11, 'ro', 0.8, 0.004)
annotate(descent3, 12, 'o', 'orange', 0.8, 0.009)

plt.xlabel('Number of iteration')
plt.ylabel(r'$\frac{1}{n}\sum(p_i - y_i)$')

plt.xlim([0, 35])
plt.ylim([-0.08, 0.04])

line1, = plt.plot(descent1.grads[:35], label=r'step = $3$', linewidth=2)
line2, = plt.plot(descent3.grads[:36], label=r'step = $20$', linewidth=2)

plt.legend(handles=[line1, line2], loc='lower right')
plt.tick_params(axis='both', which='major', labelsize=9)

plt.show()
plt.tight_layout()

reload(filters)

data = tools.load_data(limit=10000, offset=90000)

data1 = data[filters.classmates(data)]
print len(data1)

data2 = data[~filters.classmates(data)]
print len(data2)

descents = {
    'In-School': (optimize.GradientDescent(data1), {}),
    'Out-of-School': (optimize.GradientDescent(data2), {}),
}

dresults = {}
for name, (descent, kwargs),  in descents.items():
    tools.echo(name, clear=False)
    dresults[name] = descent.search_staircase(
        init_learn_rate=0.015,
        number_of_iter=20,
        **kwargs
    )

plots = []
for name, dresult in dresults.items():
    p, = dresult.plot()
    plots += [(name, p, dresult)]

if len(plots) > 1:
    gamma_delta = ' ($\gamma = {0[gamma]:.3f}, \delta = -{0[delta]:.3f}$)'
    plt.legend([item[1] for item in plots],
               [n + gamma_delta.format(r.best) for n, p, r in plots])

max_size = 100000
slices = 7
descents_10 = (
    ('Lakes', lambda d: filters.place_type(d, 'lake') & filters.for_staircase(d), 4),
    ('Rivers', lambda d: filters.place_type(d, 'river') & filters.for_staircase(d), 1),
    ('Mountains', lambda d: filters.place_type(d, 'mountains') & filters.for_staircase(d), 1),
)

dresults_10 = defaultdict(list)

for name, filter_fun, mul,  in descents_10:
    tools.echo(name, clear=False)

    train_data = []
    for i in range(slices):
        limit, offset = 5e5 * mul, (i * 1e6) + 5e5 + (5e5 * mul)
        df = tools.load_data(limit=limit, offset=offset, echo_loaded=False)
        df = df[filter_fun(df)][:max_size]
        train_data.append(df.copy())
        tools.echo('[{}]: Loaded {} answers.'.format(i, len(df)), clear=False)

    tools.echo('Data loaded.', clear=False)

    results_classmates = []
    for i in range(slices):
        descent = optimize.GradientDescent(train_data[i])
        res = descent.search_staircase(init_learn_rate=0.02, number_of_iter=15,
                                       echo_iterations=False)
        dresults_10[name].append(res)
        tools.echo('[{}]: done!'.format(i), clear=False)

def get_gamma_delta(descent_results):
    gamma_std = np.std([res.gammas[-1] for res in descent_results])
    delta_std = np.std([res.deltas[-1] for res in descent_results])
    gamma_mean = np.mean([res.gammas[-1] for res in descent_results])
    delta_mean = np.mean([res.deltas[-1] for res in descent_results])
    return {
        'std': [gamma_std, delta_std],
        'avg': [gamma_mean, -delta_mean],
    }

def prepare_plot_data(descent_results):
    x_matrix = []
    y_matrix = []
    for res in descent_results:
        stairs = sorted(res.staircases[-1].items(), key=lambda x: x[0])
        staircase_times = res.model.metadata['staircase_times']

        xi_axis = [np.mean(staircase_times[i]) for i in res.intervals]
        yi_axis = [value for interval, value in stairs]

        x_matrix.append(xi_axis)
        y_matrix.append(yi_axis)

    x_axis = []
    y_axis = []
    e_vals = []
    for i in range(len(x_matrix[0])):
        x_axis += [np.mean([x_matrix[j][i] for j in range(len(x_matrix))])]
        y_axis += [np.mean([y_matrix[j][i] for j in range(len(x_matrix))])]
        e_vals += [np.std([y_matrix[j][i] for j in range(len(x_matrix))])]
    
    return x_axis, y_axis, e_vals

plots = []
labels = []

fig = plt.figure(num=None, figsize=(7, 4), dpi=120)
ax = plt.subplot(111)

lines = ['o-', 's-', '^-']

for i, (name, results_10) in enumerate(dresults_10.items()):
    x_axis, y_axis, e_vals = prepare_plot_data(results_10)
    if len(dresults_10) == 1:
        ax.errorbar(x_axis, y_axis, e_vals,
                    ecolor='orange', elinewidth='2',
                    linestyle='--', linewidth='2',
                    capthick='2', capsize=4,
                    color='#02A5F4', marker='o')
    p, = ax.plot(x_axis, y_axis, lines[i % 3], label=name)
    plots.append(p)
    labels.append(name)
    
    tools.echo(name, clear=False)
    tools.echo('x: {}'.format([round(x, 1) for x in x_axis]), clear=False)
    tools.echo('y: {}'.format([round(y, 3) for y in y_axis]), clear=False)
    
    gamma_delta = get_gamma_delta(results_10)
    std_msg = 'std: gamma={:.3f}, delta={:.3f}'
    avg_msg = 'avg: gamma={:.3f}, delta={:.3f}'
    tools.echo(std_msg.format(*gamma_delta['std']), clear=False)
    tools.echo(avg_msg.format(*gamma_delta['avg']), clear=False)
    
    x_pos, y_pos = x_axis[5], y_axis[5]
    info = [round(x, 3) for x in gamma_delta['avg']]
    plt.annotate(r'$\gamma={0}$, $\delta={1}$'.format(*info),
                 xy=(x_pos, y_pos), xycoords='data', size='small',
                 xytext=(x_pos, y_pos + 0.15), textcoords='data',
                 bbox=dict(boxstyle="round", fc="w", linewidth=1))

plt.xscale('log')
plt.xlabel('Time from previous attempt in seconds')
plt.ylabel('Increase in memory activation')
plt.xlim([25, 1e6])

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
legend = ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), prop={'size': 12})
legend.get_frame().set_linewidth(1)

plt.show()
plt.tight_layout()
plt.subplots_adjust(right=0.73)  # adjust for the legend to fit

descent = optimize.GradientDescent(data)

r = descent.search_staircase(init_gamma=-1, init_delta=1, number_of_iter=1)

frame, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), dpi=160)

ax1.plot(r.model.metadata['gammas'], '-',
         r.model.metadata['deltas'], '-')
ax1.set_xlabel('Number of answers')
ax1.set_ylabel('Value of $\gamma$ and $\delta$')

ax2.plot([x[0] for x in r.model.metadata['rmse']],
         [x[1] for x in r.model.metadata['rmse']],
         'g-^')
ax2.set_xlabel('Number of answers')
ax2.set_ylabel('RMSE')

ax1.text(9000, 3, r'$\gamma$', fontsize=16, color='#02A5F4')
ax1.text(9000, -0.8, r'$\delta$', fontsize=16, color='orange')
#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')

plt.tight_layout()



