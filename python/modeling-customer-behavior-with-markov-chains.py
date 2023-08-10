# import modules
import itertools as it
from collections import Counter

import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

# settings for plots
get_ipython().magic('matplotlib inline')
sns.set(font_scale=1.3)
sns.set_palette(sns.husl_palette(3))
green = sns.husl_palette(3)[1]
blue = sns.husl_palette(3)[2]

# load data and set column labels
transactions = pd.read_table('purchases.txt', header=None)
transactions.columns = ['user_id', 'value', 'date']

# examine data
transactions.head()

# examine data
transactions.info()

## Deal with datetime issues

# convert date to datetime
transactions['date'] = pd.to_datetime(transactions.date)

# set date as the index
transactions = transactions.set_index('date')

# convert dates to years
transactions = transactions.to_period('A')

# reset index
transactions = transactions.reset_index()

# examine descriptive statistics
transactions.describe()

# get purchase frequency matrix
freq = pd.crosstab(transactions.user_id, transactions.date)

# examine purchase frequency matrix
freq.head()

# encode frequency segments
for y in freq.columns:
    freq.loc[:, y] = freq.loc[:, y].apply(lambda x: 2 if x > 1 else x)

freq.head()

# get frequency states
F = freq.values.copy()
n, m = F.shape
for i in range(n):
    for j in range(m):
        if (m != 0) and F[i, j] == 0:
            F[i, j] = F[i, j-1]

np.unique(F)

# get purchase value matrix and examine
mon = pd.crosstab(transactions.user_id, transactions.date, transactions.value, aggfunc=np.sum)
mon.describe()

mon.head()

# get monetary states
M = mon.values.copy()
M[np.isnan(M)] = 0
M[np.where(np.logical_and(M>0, M<30))] = 1
M[np.where(M>=30)]=2
for i in range(n):
    for j in range(m):
        if j > 0 and M[i,j] == 0:
            M[i,j] = M[i, j-1]

np.unique(M)

# get recency states
R = freq.values.copy()
for i in range(n):
    for j in range(m):
        if (j == 0) and (R[i, j] > 0):
            R[i, j] = 1
        elif j > 0:
            if R[i, j] > 0:
                R[i, j] = 1
            elif (R[i, j-1] > 0):
                R[i, j] = R[i, j-1] + 1
np.unique(R)

# examine R
pd.DataFrame(R).head()

# initialize hashmap
state_map = dict()

# get possible states for m, f, r
ms = [0, 1, 2]
fs = [0, 1, 2]
rs = list(range(11))

# get cartesian product of m, f, r states
states = it.product(ms, fs, rs)

# assign inactive states to 0
for state in states:
    if state[0] == 0 or state[1] == 0:
        state_map[state] = 0

# assign states 1-21 to customers who have made purchases but not yet churned
state = 1
for r in range(1, 12):
    state_tuple = (2, 2, r)
    state_map[state_tuple] = state
    state += 1
    state_tuple = (2, 1, r)
    state_map[state_tuple] = state
    state += 1
    state_tuple = (1, 2, r)
    state_map[state_tuple] = state
    state += 1
    state_tuple = (1, 1, r)
    state_map[state_tuple] = state
    state += 1

# assign state 21 to the absorbing state (churn, recency greater than five)
for state in state_map.keys():
    if state[2] > 5:
        state_map[state] = 21

# examine state_map values
unique_states = np.unique(list(state_map.values()))
unique_states

# collect M, F, R into a single array
MFR = np.zeros((n, m, 3))
MFR[:, :, 0] = M.copy()
MFR[:, :, 1] = F.copy()
MFR[:, :, 2] = R.copy()

# initialize state matrix
S = np.zeros((n, m), dtype=int)
for i in range(n):
    for j in range(m):
        # each entry of S is the corresponding tuple from MFR, passed through the state_map
        S[i, j] = int(state_map[tuple(MFR[i, j, :].astype(int))])

# examine S
pd.DataFrame(S).head()

# get num states
n_states = len(unique_states)

# initialize transition array. Each slice is a 22x22 transition matrix, one for each transition between periods
T_freq = np.zeros((n_states, n_states, 10), dtype=int)

for p in range(10):
    for r in range(n):
        i = S[r, p]
        j = S[r, p+1]
        T_freq[i, j, p] += 1
        
# examine one slice
T_freq[1:10, 1:10, 3]

# sum over all periods
T_freq_total = T_freq.sum(axis=2)

# examine totals
T_freq_total[1:10, 1:10]

# get number of churned customers who returned
churn_returns = T_freq_total[-1, 1:][:-1].sum()
print('churn returns: {}'.format(churn_returns))

# get total transitions
total_transitions = T_freq_total[1:, 1:].sum()
print('total transitions: {}'.format(total_transitions))

# percent of purchases missed, assuming churners wouldn't return
# if they weren't marketed to
percent_missed_purchases = churn_returns / total_transitions
print('percent missed purchases: {}'.format(percent_missed_purchases))

# initialze transition probability matrix
T_prob = np.zeros(T_freq_total.shape)
# populate values
for r in range(n_states):
    T_prob[r, :] = T_freq_total[r, :]/T_freq[r, :].sum()
    
# convert the last row into an absorbing state
T_prob[-1, :] = 0
T_prob[-1, -1] = 1

# examine transition matrix    
T_prob[1:9, 1:9].round(3)

# get purchase amounts for purchasing states
rev_2_2 = mon.values.copy()[np.logical_and(mon.values >= 30, freq.values > 1)]
rev_2_1 = mon.values.copy()[np.logical_and(mon.values >= 30, freq.values == 1)]
rev_1_2 = mon.values.copy()[np.logical_and(mon.values < 30, freq.values > 1)]
rev_1_1 = mon.values.copy()[np.logical_and(mon.values < 30, freq.values == 1)]

# get rewards
r22 = rev_2_2.mean()
r22

r21 = rev_2_1.mean()
r21

r12 = rev_1_2.mean()
r12

r11 = rev_1_1.mean()
r11

# set marketing cost and discount rate
cost = 10
d = .1
discount = 1/(1+d)
discount

# initialize reward matrix
r = np.zeros(21)
# set rewards for recency 1 states
r[:4] = np.array([r22, r21, r12, r11]) - cost

# set rewards for higher recency states
r[4:-1] = -cost
r

def clv(T, R, d):
    # get dimensions of T
    n = T.shape[0]
    # calculate CLV
    return np.linalg.inv(np.eye(n) - (1/(1+d))*T).dot(R)

CLV = clv(T_prob[1:, 1:], r, d)

CLV

policy = np.zeros(21)
policy[CLV > 0] = 1
policy[CLV <= 0] = 0
# this is the final policy
policy

# states that should not be marketed to
np.argwhere(policy == 0) + 1

# get counts for all states for each period
state_counts = np.zeros((22, 11), dtype=int)
for p in range(11):
    counts = Counter(S[:, p])
    for state in range(22):
        state_counts[state, p] = counts[state]

state_counts[:5, :5]

# get number of written-off customers who returned
policy_write_off_returns = T_freq_total[1:, 1:5][np.argwhere(policy == 0)].sum()
print('written off returns: {}'.format(policy_write_off_returns))

# get total transitions
total_transitions = T_freq_total[1:, 1:].sum()
print('total transitions: {}'.format(total_transitions))

# percent of purchases missed, assuming customers wouldn't return
# if they weren't marketed to
percent_missed_purchases = policy_write_off_returns / total_transitions
print('percent missed purchases: {:.4f}'.format(percent_missed_purchases))

# get number of first-time purchasers for each year
n_new_purch = -pd.DataFrame(state_counts[0, :]).diff().values[1:]
n_new_purch

# examine new customers for each year
sns.plt.plot(n_new_purch)
sns.plt.title('first-time purchasers')
sns.plt.xlabel('period')
sns.plt.ylabel('# customers');

# get average number of new customers each year
mu = n_new_purch.mean()
sigma = n_new_purch.std()

def new_customers():
    return sigma * np.random.randn() + mu

print('μ: {}\nσ:  {:.1f}'.format(mu, sigma))

# plot histogram of new customers
X = stats.norm(mu, sigma)
x = np.arange(500, 3500)
y = X.pdf(x)
sns.plt.plot(x, y)
sns.distplot(n_new_purch,
             hist=False,
             rug=True,
             rug_kws=dict(linewidth=3),
             color=blue)
sns.plt.title('first-time purchases')
sns.plt.xlabel('# customers')
sns.plt.ylabel('density')
sns.plt.legend(['model', 'historical']);

# get proportional probabilities for new customers entering the system
new_cust_p = T_prob[0, 1:5]
new_customer_prop = new_cust_p / new_cust_p.sum()
new_customer_prop

## refactor transition matrix and reward vectors to allow new customers to enter the system

T_prob_proj = T_prob.copy()
T_prob_proj[0, 0] = 0
T_prob_proj[0, 1:5] = new_customer_prop

# reward vector, no implementation of policy
r_proj_no_policy = np.zeros(22)
r_proj_no_policy[1:] = r.copy()

# reward vector, with policy implementation
r_proj_policy = np.zeros(22)
r_proj_policy[1:] = r.copy()*policy.copy()

## project 10 years of revenue, 10000 trials

# set periods to project
y = 10
num_trials = 10000
trials_no_policy = np.zeros((num_trials, y))
trials_policy = np.zeros((num_trials, y))

for trial in range(num_trials):
    # initialize customer distribution
    customer_dist = state_counts.copy()[:, -1]
    for year in range(y):
        customer_dist[0] += new_customers()
        customer_dist = np.dot(customer_dist, T_prob_proj)
        # without implementing policy (churn at 6 periods)
        trials_no_policy[trial, year] = np.dot(customer_dist, r_proj_no_policy)*discount**(year+1)
        # with the recommended policy
        trials_policy[trial, year] = np.dot(customer_dist, r_proj_policy)*discount**(year+1)

# get mean and std for all trials
no_pol_mu = trials_no_policy.mean(axis=0)
no_pol_sig = trials_no_policy.std(axis=0)

pol_mu = trials_policy.mean(axis=0)
pol_sig = trials_policy.std(axis=0)

# calculate and plot means and confidence intervals
no_pol_up_ci = no_pol_mu + 1.96*no_pol_sig
no_pol_down_ci = no_pol_mu - 1.96*no_pol_sig
sns.plt.plot(no_pol_mu)
sns.plt.fill_between(np.arange(10),
                     no_pol_up_ci,
                     no_pol_down_ci,
                     alpha=.2)

pol_up_ci = pol_mu + 1.96*pol_sig
pol_down_ci = pol_mu - 1.96*pol_sig
sns.plt.plot(pol_mu)
sns.plt.fill_between(np.arange(10),
                     pol_up_ci,
                     pol_down_ci,
                     alpha=.2,
                     facecolor=green)

sns.plt.xlabel('years')
sns.plt.ylabel('net income')
sns.plt.title('Projected Net Income Over 10 Years')
sns.plt.legend(['without policy', 'policy', '95% CI'], bbox_to_anchor=(1.35, 1));

# check output histograms
sns.distplot(trials_no_policy[:, 5], bins=50)
sns.distplot(trials_policy[:, 5], bins=50);

## cumulative revenue over 10 years

# get mean and std
no_pol_cum = trials_no_policy.cumsum(axis=1)
no_pol_cum_mu = no_pol_cum.mean(axis=0)
no_pol_cum_sig = no_pol_cum.std(axis=0)

pol_cum = trials_policy.cumsum(axis=1)
pol_cum_mu = pol_cum.mean(axis=0)
pol_cum_sig = pol_cum.std(axis=0)

# calculate confidence intervals
no_pol_cum_up_ci = no_pol_cum_mu + 1.96*no_pol_cum_sig
no_pol_cum_down_ci = no_pol_cum_mu - 1.96*no_pol_cum_sig
# plot
sns.plt.plot(no_pol_cum_mu)
sns.plt.fill_between(np.arange(10),
                     no_pol_cum_up_ci,
                     no_pol_cum_down_ci,
                     alpha=.2)

# calculate confidence intervals
pol_cum_up_ci = pol_cum_mu + 1.96*pol_cum_sig
pol_cum_down_ci = pol_cum_mu - 1.96*pol_cum_sig
# plot
sns.plt.plot(pol_cum_mu)
sns.plt.fill_between(np.arange(10),
                     pol_cum_up_ci,
                     pol_cum_down_ci,
                     alpha=.2,
                     facecolor=green)

# labels
sns.plt.xlabel('years')
sns.plt.ylabel('net income')
sns.plt.title('Projected Net Income Over 10 Years\n(Cumulative)')
sns.plt.legend(['without policy', 'policy', '95% CI'], bbox_to_anchor=(1.35, 1));

# total revenue over 10 years (no policy)
r_n = no_pol_cum_mu[-1]
r_n_up_ci = no_pol_cum_mu[-1] + 1.96*no_pol_cum_sig[-1]
r_n_down_ci = no_pol_cum_mu[-1] - 1.96*no_pol_cum_sig[-1]

# total revenue over 10 years (with policy)
r_p = pol_cum_mu[-1]
r_p_up_ci = pol_cum_mu[-1] + 1.96*pol_cum_sig[-1]
r_p_down_ci = pol_cum_mu[-1] - 1.96*pol_cum_sig[-1]

print('''With Policy
-------------------------------
projected net income: {:.2f}
95% CI [{:.2f}, {:.2f}]

Without Policy
-------------------------------
projected net income: {:.2f}
95% CI [{:.2f}, {:.2f}]'''.format(r_p, r_p_down_ci, r_p_up_ci,
                                  r_n, r_n_down_ci, r_n_up_ci))

# comparison
sns.plt.xlabel('total net income')
sns.plt.title('Projected Net Income Over 10 Years')
sns.barplot(x=[r_n, r_p], y=['no policy', 'policy'], orient='h');

# policy savings
savings = pol_cum - no_pol_cum
savings_mu = savings.mean(axis=0)[-1]
savings_sig = savings.std(axis=0)[-1]
savings_up_ci = savings_mu + 1.96*savings_sig
savings_down_ci = savings_mu - 1.96*savings_sig

print('''Estimated savings over 10 years: {:.2f}
95% CI: [{:.2f}, {:.2f}]'''.format(savings_mu, savings_down_ci, savings_up_ci))

# get count of marketing targets for each historical period
cost_count_no_policy = (S > 0).astype(int).sum(axis=0)
cost_count_with_policy = np.logical_and(S < 11,
                                        np.logical_and(S > 0,
                                                       np.logical_and(S != 7, 
                                                                      S != 8)))
cost_count_with_policy = cost_count_with_policy.astype(int).sum(axis=0)
# get total costs per period
costs_no_policy = cost_count_no_policy*cost
costs_with_policy = cost_count_with_policy*cost

# subtract marketing costs from historical revenue
hist_rev_no_policy  = mon.sum().values - costs_no_policy

# implementing the policy means missing out on purchases by returning churners
# so adjust the historical revenue accordingly
hist_rev_with_policy  = (1 - percent_missed_purchases)*mon.sum().values -                        costs_with_policy

sns.plt.plot(hist_rev_no_policy[1:])
sns.plt.plot(hist_rev_with_policy[1:])
sns.plt.xlabel('years')
sns.plt.ylabel('revenue')
sns.plt.title('Historical Net Income')
sns.plt.legend(['without policy', 'with policy'], loc=2);

# estimated revenue if the policy had been implemented on
# historical data
rh_n = hist_rev_no_policy.sum()
rh_p = hist_rev_with_policy.sum()

sns.plt.xlabel('total net income')
sns.plt.title('Historical Net Income Estimates')
sns.barplot(x=[rh_n, rh_p], y=['no policy', 'policy'], orient='h');

# estimated savings
print('estimated savings on last 10 years: {:.2f}'.format(rh_p - rh_n))

# set periods to project
y = 10
num_trials = 10000
# initialize simulation arrays
sim_no_policy = np.zeros((num_trials, y))
sim_policy = np.zeros((num_trials, y))

for trial in range(num_trials):
    ## simulate each year
    for year in range(y):
        # get current customer distribution
        customer_dist = state_counts.copy()[:, year]
        # add new first time purchasers for next period
        customer_dist[0] = new_customers()
        # get projected distribution for next period
        customer_dist = np.dot(customer_dist, T_prob_proj)

        ## calculate revenue
        # without implementing policy (churn at 6 periods)
        sim_no_policy[trial, year] = np.dot(customer_dist, r_proj_no_policy)*discount
        # with the recommended policy
        sim_policy[trial, year] = np.dot(customer_dist, r_proj_policy)*discount

# get mean and std for all trials
sim_no_pol_mu = sim_no_policy.mean(axis=0)
sim_no_pol_sig = sim_no_policy.std(axis=0)

sim_pol_mu = sim_policy.mean(axis=0)
sim_pol_sig = sim_policy.std(axis=0)        

# calculate and plot means and confidence intervals
sim_no_pol_up_ci = sim_no_pol_mu + 1.96*sim_no_pol_sig
sim_no_pol_down_ci = sim_no_pol_mu - 1.96*sim_no_pol_sig
sns.plt.plot(sim_no_pol_mu)
sns.plt.fill_between(np.arange(10), 
                     sim_no_pol_up_ci, 
                     sim_no_pol_down_ci, 
                     alpha=.2)

sim_pol_up_ci = sim_pol_mu + 1.96*sim_pol_sig
sim_pol_down_ci = sim_pol_mu - 1.96*sim_pol_sig
sns.plt.plot(sim_pol_mu)
sns.plt.fill_between(np.arange(10), 
                     sim_pol_up_ci, 
                     sim_pol_down_ci, 
                     alpha=.2, 
                     facecolor=green)

# labels
sns.plt.xlabel('years')
sns.plt.ylabel('net income')
sns.plt.title('Model Projections Over Historical Period')
sns.plt.legend(['without policy', 'policy', '95% CI'], bbox_to_anchor=(1.35, 1));

## the first simulation is projected from the first year, so its
## predictions should be compared with historical data from the 
## second year forward

sns.plt.plot(sim_no_pol_mu)
sns.plt.fill_between(np.arange(10),
                     sim_no_pol_up_ci,
                     sim_no_pol_down_ci,
                     alpha=.2)
sns.plt.plot(hist_rev_no_policy[1:], color=blue)
sns.plt.xlabel('years')
sns.plt.ylabel('net income')
sns.plt.title('Historical Data v Model')
sns.plt.legend(['model', 'historical', '95% CI'], loc=2);

# linear model comparing simulations with historical data
sns.jointplot(hist_rev_no_policy[1:], sim_no_pol_mu, kind='reg', 
              joint_kws=dict(scatter_kws=dict(color='k', alpha=.9)), 
              marginal_kws=dict(bins=5, rug=True, 
                                rug_kws=dict(color = 'k', linewidth=3)))
sns.plt.xticks([])
sns.plt.yticks([])
sns.plt.suptitle('Model Projections v Historical Data', y=1.01)
sns.plt.xlabel('historical')
sns.plt.ylabel('no policy model');

