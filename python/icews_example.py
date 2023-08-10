import numpy as np
import numpy.random as rn
get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pgds import PGDS

data_dict = np.load('../dat/icews/undirected/2003-D/data.npz')

Y_TV = data_dict['Y_TV']          # observed TxV count matrix
(T, V) = Y_TV.shape
print 'T = %d time steps' % T
print 'V = %d features' % V

dates_T = data_dict['dates_T']    # time steps are days in 2003
print 'First time step: %s' % dates_T[0]
print 'Last time step: %s' % dates_T[-1]

labels_V = data_dict['labels_V']  # features are undirected edges of countries 
print 'Most active feature: %s' % labels_V[0]
print 'Least active feature: %s' % labels_V[-1]

K = 100            # number of latent components
gam = 75           # shrinkage parameter
tau = 1            # concentration parameter
eps = 0.1          # uninformative gamma parameter
stationary = True  # stationary variant of the model
steady = True      # use steady state approx. (only for stationary)
shrink = True      # use the shrinkage version
binary = False     # whether the data is binary (vs. counts)
seed = 111111      # random seed (optional)

model = PGDS(T=T, V=V, K=K, eps=eps, gam=gam, tau=tau,
             stationary=int(stationary), steady=int(steady),
             shrink=int(shrink), binary=int(binary), seed=seed)

num_itns = 1000    # number of Gibbs sampling iterations (the more the merrier)
verbose = False    # whether to print out state
initialize = True  # whether to initialize model randomly

model.fit(data=Y_TV,
          num_itns=num_itns,
          verbose=verbose,
          initialize=initialize)

state = dict(model.get_state())
Theta_TK = state['Theta_TK']  # TxK time step factors
Phi_KV = state['Phi_KV']      # KxV feature factors
Pi_KK = state['Pi_KK']        # KxK transition matrix
nu_K = state['nu_K']          # K component weights

top_k = nu_K.argmax()                          # most active component
features = Phi_KV[top_k].argsort()[::-1][:10]  # top 10 features in top k
print labels_V[features]

k = top_k # which component to visualize (replace with any k = 1...K to further explore)

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

# Plot time-step factors for component k
ax1.plot(Theta_TK[:, k])
xticks = np.linspace(10, Theta_TK.shape[0] - 10, num=8, dtype=int)
ax1.set_xticks(xticks)
formatted_dates = np.asarray([pd.Timestamp(x).strftime('%b %Y') for x in dates_T])
ax1.set_xticklabels(formatted_dates[xticks], weight='bold')

# Plot top N feature factors for component k
N = 15
top_N_features = Phi_KV[k].argsort()[::-1][:N]
ax2.stem(Phi_KV[k, top_N_features])
ax2.set_xticks(range(N))
ax2.set_xticklabels(labels_V[top_N_features], rotation=330, weight='bold', ha='left')

plt.show()

# Pro tip: To find interpret the components, Google the top date and top feature(s).

k = top_k                        # which component to explore

top_t = Theta_TK[:, k].argmax()  # top time-step factor
top_date = dates_T[top_t]        # date of top time-step factor (top date)
formatted_date = pd.Timestamp(top_date).strftime('%B %d %Y')
print 'Top date: %s' % formatted_date

N = 3
top_vs = Phi_KV[k, :].argsort()[::-1][:N]   # top N features
top_features = labels_V[top_vs]
print 'Top N features: %s' % top_features

query = formatted_date + ' ' + top_features[0].replace('--', ' ')
print "Example Google query: '%s'" % query

from run_pgds import mre, mae

S = 2  # how many final time-steps to hold out completely (for forecasting)

data_SV = Y_TV[-S:]    # hold out final S time steps (to forecast)
train_TV = Y_TV[:-S]   # training data matrix is everything up to t=S
(T, V) = train_TV.shape 

percent = 0.1  # percent of training data to mask (for smoothing)
mask_TV = (rn.random(size=(T, V)) < percent).astype(bool)  # create a TxV random mask 

masked_train_TV = np.ma.array(train_TV, mask=mask_TV)

K = 100            # number of latent components
gam = 75           # shrinkage parameter
tau = 1            # concentration parameter
eps = 0.1          # uninformative gamma parameter
stationary = True  # stationary variant of the model
steady = True      # use steady state approx. (only for stationary)
shrink = True      # use the shrinkage version
binary = False     # whether the data is binary (vs. counts)
seed = 111111      # random seed (optional)

model = PGDS(T=T, V=V, K=K, eps=eps, gam=gam, tau=tau,
             stationary=int(stationary), steady=int(steady),
             shrink=int(shrink), binary=int(binary), seed=seed)

num_itns = 100     # number of Gibbs sampling iterations (the more the merrier)
verbose = False     # whether to print out state
initialize = True  # whether to initialize model randomly

model.fit(data=masked_train_TV,
          num_itns=num_itns,
          verbose=verbose,
          initialize=initialize)

pred_TV = model.reconstruct()  # reconstruction of the training matrix

train_mae = mae(train_TV[~mask_TV], pred_TV[~mask_TV])
train_mre = mre(train_TV[~mask_TV], pred_TV[~mask_TV])

smooth_mae = mae(train_TV[mask_TV], pred_TV[mask_TV])
smooth_mre = mre(train_TV[mask_TV], pred_TV[mask_TV])

print 'Mean absolute error on reconstructing observed data: %f' % train_mae
print 'Mean relative error on reconstructing observed data: %f' % train_mre
print 'Mean absolute error on smoothing missing data: %f' % smooth_mae
print 'Mean relative error on smoothing missing data: %f' % smooth_mre

pred_SV = model.forecast(n_timesteps=S)  # forecast of next S time steps

forecast_mae = mae(data_SV, pred_SV)
forecast_mre = mre(data_SV, pred_SV)

print 'Mean absolute error on forecasting: %f' % forecast_mae
print 'Mean relative error on forecasting: %f' % forecast_mre

initialize = False  # run some more iterations---i.e., don't reinitialize

model.fit(data=masked_train_TV,
          num_itns=num_itns,
          verbose=verbose,
          initialize=initialize)

new_pred_TV = model.reconstruct()  # reconstruction of the training matrix from new sample
avg_pred_TV = (pred_TV + new_pred_TV) / 2.  # compute avg. reconstruction

train_mae = mae(train_TV[~mask_TV], avg_pred_TV[~mask_TV])
train_mre = mre(train_TV[~mask_TV], avg_pred_TV[~mask_TV])

smooth_mae = mae(train_TV[mask_TV], avg_pred_TV[mask_TV])
smooth_mre = mre(train_TV[mask_TV], avg_pred_TV[mask_TV])

print 'Mean absolute error on reconstructing observed data: %f' % train_mae
print 'Mean relative error on reconstructing observed data: %f' % train_mre
print 'Mean absolute error on smoothing missing data: %f' % smooth_mae
print 'Mean relative error on smoothing missing data: %f' % smooth_mre

new_pred_SV = model.forecast(n_timesteps=S)  # forecast of next S time steps from new sample
avg_pred_SV = (pred_SV + new_pred_SV) / 2.   # compute avg. forecast

forecast_mae = mae(data_SV, avg_pred_SV) 
forecast_mre = mre(data_SV, avg_pred_SV)

print 'Mean absolute error on forecasting: %f' % forecast_mae
print 'Mean relative error on forecasting: %f' % forecast_mre

