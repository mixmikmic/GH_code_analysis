import numpy as np
from scipy import io
import sys
import os
from ucrp import UniformConstantRebalancedPortfolio
from ubah import UniformBuyAndHoldPortfolio
from util import load_matlab_sp500_data
from expert_pool import ExpertPool
from olmar import OLMAR
from rmr import RMR
from nonparametric_markowitz import NonParametricMarkowitz
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import matplotlib.patches as mpatches

x_range = range(0, numDays)
plt.plot(value_op_seq_npm, color='#25B009')
plt.plot(value_op_seq_rmr, color='red')
plt.plot(value_op_seq_pool, color='#9409B0')

pool_patch = mpatches.Patch(color='#9409B0', label='Pool')
npm_patch = mpatches.Patch(color='#25B009', label='Mark')
rmr_patch = mpatches.Patch(color='red', label='RMR')
#plt.legend(handles=[pool_patch, npm_patch, rmr_patch])
plt.legend(('Mark','RMR', 'Pool'), loc=(1.025,0.5))

plt.xlim([0, numDays])
plt.ylim([0.4, 3.5])
plt.xlabel("Day")
plt.ylabel("Value at Open (Dollars)")
plt.show()

# Load test data
directory = 'data/portfolio_train.mat'
test_data = load_matlab_sp500_data(directory)
num_days, numStocks = test_data.get_cl().shape
past_results_dir = 'train_results/' # Directory containing parameters learned during training
train_directory = 'data/portfolio_train.mat'
train_data = load_matlab_sp500_data(train_directory)

def get_results(portfolio):
    b_history = portfolio.b_history
    b_history = np.concatenate((b_history, np.reshape(portfolio.b, (-1,1))), axis=1)
    dollars_history = portfolio.get_dollars_history()
    return_seq = np.log(dollars_history[1:] / dollars_history[:-1])
    annual_return = 252 * np.mean(return_seq)
    sharpe = np.sqrt(252) * np.mean(return_seq) / np.std(return_seq)
    return b_history, dollars_history, annual_return, sharpe

def plot_dollars(dollars_li, colors=None, title=None, leg_titles=[], ylims=None):
    num_curves = len(dollars_li)
    if colors is None:
        colors = num_curves * ['blue']
    
    for c, dollars in zip(colors, dollars_li):
        plt.plot(dollars, color=c)

    num_days = len(dollars)
    plt.xlim([0, num_days])
    if ylims:
        plt.ylim(ylims)
    else:
        plt.ylim([0.4, 3.5])
    plt.xlabel("Day")
    plt.ylabel("Value at Open (Dollars)")
    
    if title:
        plt.title(title)
    
    if len(leg_titles) > 0:
        plt.legend((leg_titles), loc=(1.025,0.5))
    plt.show()

ucrp_ind = UniformConstantRebalancedPortfolio(market_data=test_data, tune_interval=None, verbose=True, 
                                         past_results_dir=past_results_dir+'UCRP/', repeat_past=True)
ucrp_ind.run()
ucrp_b_history, ucrp_dollars_history, ucrp_anret, ucrp_sharpe = get_results(ucrp_ind)
print 'Final dollars: ', str(ucrp_dollars_history[-1])
print 'Annual Return: ', str(ucrp_anret)
plot_dollars([ucrp_dollars_history], colors=['g'])

rmr_ind = RMR(market_data=test_data, market_data_train=train_data, tune_interval=60, verbose=True, past_results_dir=past_results_dir+'RMR/', repeat_past=True)
rmr_ind.run()
rmr_b_history, rmr_dollars_history, rmr_anret, rmr_sharpe = get_results(rmr_ind)
print 'Final dollars: ', str(rmr_dollars_history[-1])
print 'Annual Return: ', str(rmr_anret)
plot_dollars([rmr_dollars_history], colors=['r'])

olmar_ind = OLMAR(market_data=test_data, market_data_train=train_data, tune_interval=20, verbose=True, past_results_dir=past_results_dir+'OLMAR/', repeat_past=True) 
olmar_ind.run()
olmar_b_history, olmar_dollars_history, olmar_anret, olmar_sharpe = get_results(olmar_ind)
print 'Final dollars: ', str(olmar_dollars_history[-1])
print 'Annual Return: ', str(olmar_anret)
plot_dollars([olmar_dollars_history], colors=['orange'])

npm_ind = NonParametricMarkowitz(market_data=test_data, market_data_train=train_data, tune_interval=None, verbose=True, past_results_dir=past_results_dir+'NPM/', repeat_past=True)
npm_ind.run()
npm_b_history, npm_dollars_history, npm_anret, npm_sharpe = get_results(npm_ind)
print 'Final dollars: ', str(npm_dollars_history[-1])
print 'Annual Return: ', str(npm_anret)
plot_dollars([npm_dollars_history], colors=['g'])

# Test Performance of Expert Pool
ucrp = UniformConstantRebalancedPortfolio(market_data=test_data, tune_interval=None, verbose=True, 
                                         past_results_dir=past_results_dir+'UCRP/', repeat_past=True)
olmar = OLMAR(market_data=test_data, market_data_train=train_data, tune_interval=20, verbose=True, 
              past_results_dir=past_results_dir+'OLMAR/', repeat_past=True) 
rmr = RMR(market_data=test_data, market_data_train=train_data, tune_interval=60, verbose=True, 
          past_results_dir=past_results_dir+'RMR/', repeat_past=True)
npm = NonParametricMarkowitz(market_data=test_data, market_data_train=train_data, tune_interval=None, 
                             verbose=True, past_results_dir=past_results_dir+'NPM/', repeat_past=True)
pool = ExpertPool(market_data=test_data, experts=[ucrp, olmar, rmr, npm], verbose=True, ew_eta=0.2, windows=[10]) 
pool.run()

pool_b_history, pool_dollars_history, pool_anret, pool_sharpe = get_results(pool)
print 'Final dollars: ', str(pool_dollars_history[-1])
print 'Annual Return: ', str(pool_anret)
plot_dollars([pool_dollars_history], colors=['g'])
weights_history = pool.weights_history

plot_dollars([ucrp_dollars_history, rmr_dollars_history, 
              olmar_dollars_history, npm_dollars_history, pool_dollars_history], 
             colors=['b', 'r', 'orange', 'g', 'purple'], 
             leg_titles=['UCRP','RMR', 'OLMAR','NPM','EPEx'],
            title='W=1, L1=10, Eta=0.2')

plt.plot(weights_history[:,0])
plt.show()

plt.plot(weights_history[:,1])
plt.show()

plt.plot(weights_history[:,2])
plt.show()

plt.plot(weights_history[:,3])
plt.show()
"""
plt.bar(range(0, numDays),weights_history[:,3], color='blue')
plt.xlim([0,numDays])
plt.show()
"""

eta_range = np.arange(0.1, 1.1, 0.1)
eta_range = np.append(eta_range, 3.0)
eta_range = np.append(eta_range, 5.0)
eta_range = np.append(eta_range, 10.0)
pools = []
num_experts = 4
num_days = test_data.get_cl().shape[0]
weights_histories = np.zeros(shape=(len(eta_range), num_days, num_experts))
for (i, eta) in enumerate(eta_range):
    ucrp = UniformConstantRebalancedPortfolio(market_data=test_data, tune_interval=None, verbose=True, 
                                         past_results_dir=past_results_dir+'UCRP/', repeat_past=True)
    olmar = OLMAR(market_data=test_data, market_data_train=train_data, tune_interval=20, verbose=True, 
                  past_results_dir=past_results_dir+'OLMAR/', repeat_past=True) 
    rmr = RMR(market_data=test_data, market_data_train=train_data, tune_interval=60, verbose=True, 
              past_results_dir=past_results_dir+'RMR/', repeat_past=True)
    npm = NonParametricMarkowitz(market_data=test_data, market_data_train=train_data, tune_interval=None, 
                                 verbose=True, past_results_dir=past_results_dir+'NPM/', repeat_past=True)
    pools.append(ExpertPool(market_data=test_data, 
                            experts=[ucrp, olmar, rmr, npm], 
                            verbose=False, silent=True, ew_eta=eta, windows=[10])) 
    pools[i].run()
    weights_histories[i,:,:] = pools[i].weights_history
    #pool_b_history, pool_dollars_history, pool_anret, pool_sharpe = get_results(pool)
    #plot_dollars([pool_dollars_history], colors=['g'])

thresholds = np.arange(0.2,1.025,0.025)
thresholds[-1] = 1.0  # avoids an issue with numpy where it treats 1.0 differently

# Array storing fraction of days exceeding different threshold for each
# (eta, threshold) pair
eta_effect = np.zeros((len(eta_range), len(thresholds)))

for i,eta in enumerate(eta_range):
    cur_weights_history = weights_histories[i]
    for j, thresh in enumerate(thresholds):
        num_greater = 0  # Number of days where at least 1 weight exceeds thresh
        for weights in cur_weights_history:
            # Count how many weights are given than each threshold
            num_greater += 1.0 * int(np.max(weights) >= thresh)
        eta_effect[i, j] = num_greater / num_days

def plot_eta_effect(eta_effect_arr, thresholds, eta_range, 
                    colors=['r','blue', 'g'], leg_labels=[], num_leg_cols=1):
    if len(colors) < len(eta_range):
        colors = len(eta_range) * ['blue']
    cmap = plt.cm.jet
    min_eta = int(0)
    max_eta = int(len(eta_range)-1)

    scalarMap = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_eta, vmax=max_eta))
    plt.figure(figsize=(7,5))
    for i, (frac_greater, eta, col) in enumerate(zip(eta_effect_arr, eta_range, colors)):
        colorVal = scalarMap.to_rgba(i)
        plt.plot(thresholds, frac_greater, color=colorVal)
        
    if len(leg_labels) > 0:
        legend = plt.legend((leg_labels), loc=(0.0,-0.5), title='Eta', ncol=num_leg_cols)
        legend.get_title().set_fontsize('12')
    plt.xlim([thresholds[0], 1.0])
    plt.xlabel('Threshold', size=12)
    plt.ylabel('Fraction of Days', size=12)
    plt.show()

plot_eta_effect(eta_effect, thresholds, eta_range, leg_labels=[str(eta) for eta in eta_range], num_leg_cols=4)

print thresholds
print eta_effect[-1]

def plot_window_results(pool_dollars_histories_w, window_configs, ylims=None, 
                        title=None, num_legend_cols=3, legend_title=None, legend_loc='below', cmap=True):
    num_curves = pool_dollars_histories_w.shape[0]
    num_days = pool_dollars_histories_w.shape[1]
    
    cmap = plt.cm.gist_rainbow
    min_val = int(0)
    max_val = num_curves-1
    scalarMap = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_val, vmax=max_val))
        
    for i, dollars in enumerate(pool_dollars_histories_w):
        colorVal = scalarMap.to_rgba(i)
        plt.plot(dollars, color=colorVal)
    #for c, dollars in zip(colors, dollars_li):
    #    plt.plot(dollars, color=c)

    num_days = len(dollars)
    plt.xlim([0, num_days])
    if ylims:
        plt.ylim(ylims)
    else:
        plt.ylim([0.5, 4.0])
    plt.xlabel("Day")
    plt.ylabel("Value at Open (Dollars)")
    
    if title:
        plt.title(title)
    
    legend_labels = []
    for w in window_configs:
        if w[0] > num_days:
            legend_labels.append('All')
        else:
            legend_labels.append(','.join([str(val) for val in w]))
    #plt.legend((legend_labels), loc=(1.025,0.5))
    if legend_title is None:
        legend_title = 'Window Configs'
    
    if legend_loc == 'below':
        loc = (0.0, -0.5)
    else:
        loc = (1.025, 0.3)
    legend = plt.legend((legend_labels), loc=loc, title=legend_title, ncol=num_legend_cols)
    legend.get_title().set_fontsize('12')
    plt.show()

plot_window_results(pool_dollars_histories_wc, window_configs, title='Alpha=0.5, Eta=0.2', ylims=[0.8,5.6],
                    legend_title='Window Configurations', num_legend_cols=3, legend_loc='below', cmap=True)

window_lengths = [[4],[5]]
for w in np.arange(10, 40, 20):
    window_lengths.append([w])
window_lengths.append([100])
window_lengths.append([200])
window_lengths.append([1250])
    
num_lengths = len(window_lengths)
sharpes_w = np.zeros(num_lengths)
anret_w = np.zeros(num_lengths)
pool_dollars_histories_w = np.zeros(shape=(num_lengths,num_days))
#eta_range = np.append(eta_range, 3.0)

for (i, windows) in enumerate(window_lengths):
    ucrp = UniformConstantRebalancedPortfolio(market_data=test_data, tune_interval=None, verbose=True, 
                                             past_results_dir=past_results_dir+'UCRP/', repeat_past=True)
    olmar = OLMAR(market_data=test_data, market_data_train=train_data, tune_interval=20, verbose=True, 
                  past_results_dir=past_results_dir+'OLMAR/', repeat_past=True) 
    rmr = RMR(market_data=test_data, market_data_train=train_data, tune_interval=60, verbose=True, 
              past_results_dir=past_results_dir+'RMR/', repeat_past=True)
    npm = NonParametricMarkowitz(market_data=test_data, market_data_train=train_data, tune_interval=None, 
                                 verbose=True, past_results_dir=past_results_dir+'NPM/', repeat_past=True)
    pool = ExpertPool(market_data=test_data, experts=[ucrp, olmar, rmr, npm],
                      verbose=False, silent=True, ew_eta=0.2, windows=windows) 
    pool.run()

    pool_b_history, pool_dollars_history, pool_anret, pool_sharpe = get_results(pool)
    pool_dollars_histories_w[i] = pool_dollars_history
    anret_w[i] = pool_anret
    sharpes_w[i] = pool_sharpe

plot_window_results(pool_dollars_histories_w, window_lengths, ylims=[0.8, 3.8], title='W=1, Eta=0.2', 
                    legend_title='L (Window Length)', num_legend_cols=4, legend_loc='below')

window_configs = [[5, 5], [5, 5, 10], [5, 5, 20], [5, 10], [5, 10, 10], [10, 5], [10,10], [10, 20], [10, 30]]
num_configs = len(window_configs)
sharpes_wc = np.zeros(num_configs)
anret_wc = np.zeros(num_configs)
pool_dollars_histories_wc = np.zeros(shape=(num_configs,num_days))
#eta_range = np.append(eta_range, 3.0)

for (i, windows) in enumerate(window_configs):
    ucrp = UniformConstantRebalancedPortfolio(market_data=test_data, tune_interval=None, verbose=True, 
                                             past_results_dir=past_results_dir+'UCRP/', repeat_past=True)
    olmar = OLMAR(market_data=test_data, market_data_train=train_data, tune_interval=20, verbose=True, 
                  past_results_dir=past_results_dir+'OLMAR/', repeat_past=True) 
    rmr = RMR(market_data=test_data, market_data_train=train_data, tune_interval=60, verbose=True, 
              past_results_dir=past_results_dir+'RMR/', repeat_past=True)
    npm = NonParametricMarkowitz(market_data=test_data, market_data_train=train_data, tune_interval=None, 
                                 verbose=True, past_results_dir=past_results_dir+'NPM/', repeat_past=True)
    pool = ExpertPool(market_data=test_data, experts=[ucrp, olmar, rmr, npm],
                      verbose=False, silent=True, ew_eta=0.2, windows=windows) 
    pool.run()

    pool_b_history, pool_dollars_history, pool_anret, pool_sharpe = get_results(pool)
    pool_dollars_histories_wc[i] = pool_dollars_history
    anret_wc[i] = pool_anret
    sharpes_wc[i] = pool_sharpe

plot_window_results(pool_dollars_histories_wc, window_configs, title='Alpha=0.5, Eta=0.2', ylims=[0.8,5.6],
                    legend_title='Window Configurations', num_legend_cols=3, legend_loc='below')

plot_window_results(pool_dollars_histories_wc, window_configs, title='Alpha=0.5, Eta=0.2', ylims=[0.8,5.6],
                    legend_title='Window Configurations', num_legend_cols=3, legend_loc='below', cmap=True)



