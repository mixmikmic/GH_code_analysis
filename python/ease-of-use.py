import numpy as np, GPy, pandas as pd
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

get_ipython().magic('run load_abalone')
data.head()

# Make training and test possible, by subsetting the data randomly. 
# We will use 1/8 of the data as training set:
train_idx = np.random.choice(data.shape[0], data.shape[0]/8)
test_idx = np.setdiff1d(np.arange(data.shape[0]), train_idx)[:400]
# make sex a categorical factor, for usage in regression task
_tmp = data.copy()
sex_labels, sex_uniques = pd.factorize(data['sex'], size_hint=3)
_tmp['sex'] = sex_labels
data_train, data_test = _tmp.loc[train_idx].copy(), _tmp.loc[test_idx].copy()

# Select one input for now: we will use the diameter:
selected_input = 'length'
simple_kern = GPy.kern.Matern32(1, # How many dimension does this kernel have?
                                name=selected_input # The name of this kernel
                               )
# Try out different kernels and see how the output changes. 
# This process is called model selection, as the kernel
# represents the prior believe over function
# values of the output.

X_train = data_train.loc[:, 'sex':'shell_weight']
Y_train = data_train.loc[:, ['rings']]

X_test = data_test.loc[:, 'sex':'shell_weight']
Y_test = data_test.loc[:, ['rings']]

simple_model = GPy.models.GPRegression(X_train.loc[:,[selected_input]].values, Y_train.values, kernel=simple_kern)

simple_model

# Optimize the simple model and plot the result
simple_model.optimize(messages=1)

_ = simple_model.plot(plot_training_data=True)

def log_likelihood(m, X_test, Y_test):
    import scipy as sp
    mu, var = m.predict(X_test, full_cov=True)
    return sp.stats.multivariate_normal.logpdf(Y_test.flatten(), mu.flatten(), var).sum()

def RMSE(m, X_test, Y_test):
    mu, var = m.predict(X_test, full_cov=False)
    return np.sqrt(((mu - Y_test)**2).mean())
    

result_df = pd.DataFrame(index=['Log-likelihood', 'Root-mean-squared-error'])
result_df['simple'] = [log_likelihood(simple_model, X_test[[selected_input]].values, Y_test.values), RMSE(simple_model, X_test[[selected_input]].values, Y_test.values)]

result_df

# Create one kernel for each input dimension, except the categorical (sex) dimension:
kernels = []
for dim, name in enumerate(X_train.columns):
    if name != "sex":
        kernels.append(GPy.kern.Matern32(1, #input dim
                                         active_dims=[dim], # Dimension of X to work on
                                         name=name
                                        )
                      )
kern_numeric = reduce(lambda a,b: a+b, kernels)
kern_numeric.name = 'kernel'

kern_numeric

numerical_model = GPy.models.SparseGPRegression(X_train.values, Y_train.values, 
                                                kernel=kern_numeric, num_inducing=40)

numerical_model

numerical_model.optimize(messages=1, max_iters=3e3)

numerical_model.kern.plot_ARD()
plt.legend()

# Evaluate the log likelihood for the whole numerical model:
result_df['all numerics'] = [log_likelihood(numerical_model, X_test.values, Y_test.values),
          RMSE(numerical_model, X_test.values, Y_test.values)]

result_df

_ = data[['sex','rings']].boxplot(by='sex')

kern_coreg = GPy.kern.Coregionalize(1, # Dimensionality of coregionalize input
                                    len(sex_uniques), # How many different tasks do we have?
                                    rank=3, # How deep is the coregionaliztion,
                                    # the higher the rank, the more flexible the
                                    # coregionalization, rank >= 1
                                    active_dims=[X_train.columns.get_loc('sex')] ,
                                    # Which dimension of the input carries the task information?
                                    name = 'sex',
                                   ) * kern_numeric.copy()

full_model = GPy.models.SparseGPRegression(X_train.values, Y_train.values, kernel=kern_coreg, num_inducing=40)

full_model.optimize(messages=1, max_iters=3e3)

W, k = full_model.kern.sex.parameters
coregionalization_matrix = GPy.util.linalg.tdot(W)
GPy.util.diag.add(coregionalization_matrix, k)

fig, ax = plt.subplots()
c = ax.matshow(coregionalization_matrix, cmap=plt.cm.hot)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(sex_uniques[[0, 1, 2]], fontdict=dict(size=23))
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(sex_uniques[[0, 1, 2]], fontdict=dict(thickness='bold'))
ax.xaxis.tick_top()
plt.colorbar(c, ax=ax)
fig.tight_layout()

full_model.kern.kernel.plot_ARD()
_ = plt.legend()

# Evaluate the log likelihood for the whole numerical model with coregionalization:
result_df['coreg'] = (
    log_likelihood(full_model, X_test.values, Y_test.values),
    RMSE(full_model, X_test.values, Y_test.values)
)

result_df

result_df.loc['neg-log-likelihood'] = -result_df.loc['Log-likelihood']

_ = result_df.T[['neg-log-likelihood', 'Root-mean-squared-error']].plot(kind='bar', subplots=True, legend=False)

