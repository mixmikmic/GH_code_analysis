get_ipython().magic('run "examples/simulate_networks.py" # Modify this line to be /to/path/jf2016-tutorial/examples/simulate_networks.py')

get_ipython().magic('matplotlib inline')
# Import from skggm
import inverse_covariance as ic
from inverse_covariance import (
    QuicGraphLasso,
    QuicGraphLassoCV,
    QuicGraphLassoEBIC,
    AdaptiveGraphLasso,
    ModelAverage
)
from inverse_covariance.plot_util import trace_plot 

# Notebook wide settings
n_features = 15
adj_type = 'banded'

get_ipython().magic('matplotlib inline')
covariance, precision, adjacency = new_graph(15,.15,adj_type=adj_type,random_sign=True,seed=1)    
covariance2, precision2, adjacency2 = new_graph(15,.2,adj_type=adj_type,random_sign=True, seed=1)    

# Set up the matplotlib figure
f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10, 4));

def compare_population_parameters(covariance,precision,adjacency):
    mask = np.zeros_like(precision, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    mask[np.where(np.eye(np.shape(precision)[0]))] = True

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    adj_vmax = np.max(np.triu(adjacency,1))
    sns.heatmap(adjacency,mask=mask,cmap=cmap, vmax=adj_vmax,
                square=True, xticklabels=5, yticklabels=5,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax1)
    ax1.set_title('Adjacency of Precision Matrix')

    prec_vmax = np.max(np.triu(precision,1))
    sns.heatmap(precision,mask=mask,cmap=cmap, vmax=prec_vmax,
                square=True, xticklabels=5, yticklabels=5,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax2)
    ax2.set_title('Precision Matrix')

    cov_vmax = np.max(np.triu(covariance,1))
    sns.heatmap(covariance, mask=mask,cmap=cmap, vmax=cov_vmax,
                square=True, xticklabels=5, yticklabels=5,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax3)
    ax3.set_title('Covariance Matrix')
    
compare_population_parameters(covariance,precision,adjacency)    

get_ipython().magic('matplotlib inline')

prng = np.random.RandomState(2)
X1 = mvn(75*n_features,n_features,covariance,random_state=prng)
X2 = mvn(10*n_features,n_features,covariance,random_state=prng)
X3 = mvn(75*n_features,n_features,covariance2,random_state=prng)
X4 = mvn(10*n_features,n_features,covariance2,random_state=prng)

mask = np.zeros_like(precision, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
mask[np.where(np.eye(np.shape(precision)[0]))] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
f, ([ax1, ax2, ax5, ax7], [ax3, ax4, ax6, ax8]) = plt.subplots(2,4,figsize=(14,6));
cov_vmax = np.max(np.triu(covariance,1))
sns.heatmap(np.tril(np.cov(X1,rowvar=False),-1),mask=mask,cmap=cmap, vmax=cov_vmax,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .5},ax=ax1)
ax1.set_title('Sample Covariance Matrix, n/p=75')
sns.heatmap(np.tril(np.cov(X2,rowvar=False),-1),mask=mask,cmap=cmap, vmax=cov_vmax,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .5},ax=ax2)
ax2.set_title('Sample Covariance Matrix, n/p=10')
sns.heatmap(np.tril(sp.linalg.pinv(np.cov(X1,rowvar=False)),-1),mask=mask,cmap=cmap, vmax=cov_vmax,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .5},ax=ax3)
ax3.set_title('Sample Precision Matrix, n/p=75')
sns.heatmap(np.tril(sp.linalg.pinv(np.cov(X2,rowvar=False)),-1),mask=mask,cmap=cmap, vmax=cov_vmax,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .5},ax=ax4)
ax4.set_title('Sample Precision Matrix, n/p=10')
sns.heatmap(np.tril(np.cov(X3,rowvar=False),-1),mask=mask,cmap=cmap, vmax=cov_vmax,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .5},ax=ax5)
ax5.set_title('Sample Covariance Matrix, n/p=75')
sns.heatmap(np.tril(sp.linalg.pinv(np.cov(X3,rowvar=False)),-1),mask=mask,cmap=cmap, vmax=cov_vmax,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .5},ax=ax6)
ax6.set_title('Sample Precision Matrix, n/p=75')
sns.heatmap(np.tril(np.cov(X4,rowvar=False),-1),mask=mask,cmap=cmap, vmax=cov_vmax,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .5},ax=ax7)
ax7.set_title('Sample Covariance Matrix, n/p=10')
sns.heatmap(np.tril(sp.linalg.pinv(np.cov(X4,rowvar=False)),-1),mask=mask,cmap=cmap, vmax=cov_vmax,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .5},ax=ax8)
ax8.set_title('Sample Precision Matrix, n/p=10')

get_ipython().magic('matplotlib inline')
# show graphical lasso path 
covariance, precision, adjacency = new_graph(15,.2,adj_type=adj_type,random_sign=True,seed=1)    
prng = np.random.RandomState(2)
X = mvn(20*n_features,n_features,covariance,random_state=prng)
path = np.logspace(np.log10(0.01), np.log10(1.0), num=25, endpoint=True)[::-1]
estimator = QuicGraphLasso(lam=1.0,path=path,mode='path')
estimator.fit(X)
trace_plot(estimator.precision_, estimator.path_)

metric='log_likelihood';
covariance, precision, adjacency = new_graph(n_features,.05,adj_type='banded',random_sign=True,seed=1) # BUG:Might fail with other graphs   
prng = np.random.RandomState(2)
X = mvn(10*n_features,n_features,covariance,random_state=prng)

print 'QuicGraphLassoCV with:'
print '   metric: {}'.format(metric)
cv_model = QuicGraphLassoCV(
        cv=2, 
        n_refinements=6,
        n_jobs=1,
        init_method='cov',
        score_metric=metric)
cv_model.fit(X)
cv_precision_ = cv_model.precision_
print '   len(cv_lams): {}'.format(len(cv_model.cv_lams_))
print '   lam_scale_: {}'.format(cv_model.lam_scale_)
print '   lam_: {}'.format(cv_model.lam_)

# EBIC
gamma = .1 # gamma = 0 equivalent to BIC and gamma=.5 for ultra high dimensions
ebic_model = QuicGraphLassoEBIC(
    lam=1.0,
    init_method='cov',
    gamma = gamma)
ebic_model.fit(X)
ebic_precision_ = ebic_model.precision_
print 'QuicGraphLassoEBIC with:'
print '   len(path lams): {}'.format(len(ebic_model.path_))
print '   lam_scale_: {}'.format(ebic_model.lam_scale_)
print '   lam_: {}'.format(ebic_model.lam_)

get_ipython().magic('matplotlib inline')
f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,4));

mask = np.zeros_like(precision, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
mask[np.where(np.eye(np.shape(precision)[0]))] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)

prec_vmax = np.max(np.triu(np.abs(adjacency),1))
sns.heatmap(np.abs(adjacency),cmap=cmap, vmax=prec_vmax,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .5},ax=ax1)
ax1.set_title('True Precision, CV')

prec_vmax = np.max(np.triu(np.abs(cv_precision_),1))
sns.heatmap(np.abs(cv_precision_),cmap=cmap, vmax=prec_vmax,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .5},ax=ax2)
ax2.set_title('Selected Precision, CV')

prec_vmax = np.max(np.triu(np.abs(ebic_precision_),1))
sns.heatmap(np.abs(ebic_precision_),cmap=cmap, vmax=prec_vmax,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .5},ax=ax3)
ax3.set_title('Selected Precision, EBIC')

get_ipython().magic('matplotlib inline')
f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,4));

covariance, precision, adjacency = new_graph(15,.15,adj_type=adj_type,random_sign=True,seed=1)    
n_samples = 75*n_features
prng = np.random.RandomState(2)
X = mvn(n_samples,n_features,covariance,random_state=prng)

print 'n = {},p = {}'.format(n_samples,n_features)

def compare_init_adaptive(X,n_samples,n_features):
    # Initial Estimator
    initial_estimator = QuicGraphLassoCV(init_method='corrcoef')
    initial_estimator.fit(X)
    prec_hat = initial_estimator.precision_
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    prec_vmax = np.max(np.triu(prec_hat,1))
    sns.heatmap(initial_estimator.precision_,cmap=cmap, vmax=prec_vmax,
                square=True, xticklabels=5, yticklabels=5,
                linewidths=.5, cbar_kws={"shrink": .5},ax=ax1)
    ax1.set_title('Precision Matrix, Initial Estimator')
    # Check Average Power
    err_frob, err_supp, err_fp, err_fn, err_inf =  ae_trial(
        trial_estimator=initial_estimator,
        n_samples=n_samples, 
        n_features=n_features, 
        cov=covariance, 
        adj=adjacency, 
        random_state=np.random.RandomState(2),X=X)
    print 'Difference in sparsity: {},{}'.format(
        np.sum(np.not_equal(precision,0)), 
        np.sum(np.not_equal(initial_estimator.precision_,0))
    )
    print 'Frob Norm: {} ({}), Support Error: {}, False Pos: {}, False Neg: {}'.format(
        err_frob, err_inf,
        err_supp,
        err_fp,
        err_fn
    )   
    # Adaptive Estimator
    twostage = AdaptiveGraphLasso(estimator=initial_estimator,method='inverse')
    twostage.fit(X)
    weighted_estimator = twostage.estimator_

    prec_hat = weighted_estimator.precision_
    prec_vmax = np.max(np.triu(prec_hat,1))
    sns.heatmap(weighted_estimator.precision_,cmap=cmap, vmax=prec_vmax,
                square=True, xticklabels=5, yticklabels=5,
                linewidths=.5, cbar_kws={"shrink": .5},ax=ax2)
    ax2.set_title('Precision Matrix, Adaptive Estimator')
    print 'Difference in sparsity: {},{}'.format(
        np.sum(np.not_equal(precision,0)), 
        np.sum(np.not_equal(weighted_estimator.precision_,0))
    )
    # Check Average Power
    err_frob, err_supp, err_fp, err_fn, err_inf =  ae_trial(
        trial_estimator=weighted_estimator,
        n_samples=n_samples, 
        n_features=n_features, 
        cov=covariance, 
        adj=adjacency, 
        random_state=np.random.RandomState(2), X = X)
    print 'Frob Norm: {} ({}), Support Error: {}, False Pos: {}, False Neg: {}'.format(
        err_frob,err_inf,
        err_supp,
        err_fp,
        err_fn
    )
    print 
    prec_vmax = np.max(np.triu(precision,1))
    sns.heatmap(adjacency,cmap=cmap, vmax=prec_vmax,
                square=True, xticklabels=5, yticklabels=5,
                linewidths=.5, cbar_kws={"shrink": .5},ax=ax3)
    ax3.set_title('True Precision')

compare_init_adaptive(X,n_samples,n_features)


get_ipython().magic('matplotlib inline')
f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,4));

n_samples = 15*n_features
prng = np.random.RandomState(2)
X = mvn(n_samples,n_features,covariance,random_state=prng)
print 'n = {},p = {}'.format(n_samples,n_features)
compare_init_adaptive(X,n_samples,n_features)

get_ipython().magic('matplotlib inline')
f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,4));
covariance, precision, adjacency = new_graph(15,.4,adj_type=adj_type,random_sign=True,seed=1)    
compare_population_parameters(covariance,precision,adjacency)    

get_ipython().magic('matplotlib inline')
f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10, 4));

n_samples = 75*n_features
prng = np.random.RandomState(2)
X = mvn(n_samples,n_features,covariance,random_state=prng)
print 'n = {},p = {}'.format(n_samples,n_features)

compare_init_adaptive(X,n_samples,n_features)

get_ipython().magic('matplotlib inline')
f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10, 4));

n_samples = 20*n_features
prng = np.random.RandomState(2)
X = mvn(n_samples,n_features,covariance,random_state=prng)
print 'n = {},p = {}'.format(n_samples,n_features)

compare_init_adaptive(X,n_samples,n_features)

covariance, precision, adjacency = new_graph(15,.15,adj_type=adj_type,random_sign=True,seed=1)    
n_samples = 15*n_features
prng = np.random.RandomState(2)
X = mvn(n_samples,n_features,covariance,random_state=prng)

ensemble_estimator = ModelAverage(
            n_trials=50,
            penalization='fully-random',
            lam=0.15)
ensemble_estimator.fit(X)

covariance, precision, adjacency = new_graph(15,.3,adj_type=adj_type,random_sign=True,seed=1)    
n_samples = 15*n_features
prng = np.random.RandomState(2)
X = mvn(n_samples,n_features,covariance,random_state=prng)

ensemble_estimator2 = ModelAverage(
            n_trials=50,
            penalization='fully-random',
            lam=0.15)
ensemble_estimator2.fit(X)

get_ipython().magic('matplotlib inline')
# Plot comparison
f, (ax2, ax3) = plt.subplots(1,2, figsize=(10, 4));

stability_threshold = .5
prec_hat = ensemble_estimator.proportion_
prec_vmax = np.max(np.triu(prec_hat,1))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(prec_hat,cmap=cmap, vmax=prec_vmax,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .5},ax=ax2)
ax2.set_title('Stability Matrix, deg/p=.15')
print 'Difference in sparsity: {},{}'.format(
    np.sum(np.not_equal(precision,0)), 
    np.sum(np.not_equal(prec_hat>stability_threshold,0))
)
err_fp, err_fn = _false_support(precision,np.greater(ensemble_estimator.proportion_,stability_threshold))
print 'Support Error:, False Pos: {}, False Neg: {}'.format(
    err_fp,
    err_fn
)
 
prec_hat = ensemble_estimator2.proportion_
prec_vmax = np.max(np.triu(prec_hat,1))
sns.heatmap(prec_hat,cmap=cmap, vmax=prec_vmax,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .5},ax=ax3)
ax3.set_title('Stability Matrix, deg/p=.3')
print 'Difference in sparsity: {},{}'.format(
    np.sum(np.not_equal(precision,0)), 
    np.sum(np.not_equal(prec_hat>stability_threshold,0))
)
err_fp, err_fn = _false_support(precision,np.greater(ensemble_estimator2.proportion_,stability_threshold))
print 'Support Error:, False Pos: {}, False Neg: {}'.format(
    err_fp,
    err_fn
)



