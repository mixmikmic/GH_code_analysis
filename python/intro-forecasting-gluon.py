import mxnet as mx
import mxnet.ndarray as nd
from mxnet import autograd
from mxnet import gluon
from matplotlib import pyplot as plt
import numpy as np

mx.random.seed(42)
ctx = mx.cpu()

# periodicity of the seasonality
seasonal_period = 7
# dimensions
train_length = 100
pred_length = 50
smpl_length = train_length + pred_length
# set parameter values
mu = 1
beta = 0.1
gamma = 1
sigma = 0.5
# generate the data
X = nd.arange(smpl_length).reshape((smpl_length,1)) 
S = nd.sin(2*np.pi*nd.arange(smpl_length)/seasonal_period).reshape((smpl_length,1)) 
noise = nd.random_normal(shape=(smpl_length,1))
y= mu + beta*X + gamma*S + sigma*noise

features = nd.concat(X,S)

batch_size = 4
season_train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(features, y),
                                      batch_size=batch_size, shuffle=True)

def plot_components(data, trend, seasonal, noise):
    plt.plot(data.asnumpy(), color="r")
    plt.plot(trend.asnumpy(), color="g")
    plt.plot(seasonal.asnumpy(), color="b")
    plt.plot(noise.asnumpy(), color="black")
    plt.legend(["Data", "Trend", "Seasonality", "Noise"]);

get_ipython().magic('matplotlib inline')
plot_components(y,beta*X,mu + gamma*S,sigma*noise)

season_ctx = mx.cpu()
season_net = gluon.nn.Sequential()

with season_net.name_scope():
    season_net.add(gluon.nn.BatchNorm(axis=1, center=False, scale=True))
    season_net.add(gluon.nn.Dense(1, in_units=2))
    
season_net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=season_ctx)

square_loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(season_net.collect_params(), 'sgd', {'learning_rate': 0.01})

print(season_net.collect_params())

epochs = 30
smoothing_constant = .01
moving_loss = 0
niter = 0
loss_seq = []

for e in range(epochs):
    for i, (data, label) in enumerate(season_train_data):
        data = data.as_in_context(season_ctx)
        label = label.as_in_context(season_ctx)
        with autograd.record():
            output = season_net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        
        ##########################
        #  Keep a moving average of the losses
        ##########################
        niter +=1
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss

        # correct the bias from the moving averages
        est_loss = moving_loss/(1-(1-smoothing_constant)**niter)
        loss_seq.append(est_loss)
    if e % 10 ==0:
        print("Epoch %s. Moving avg of MSE: %s" % (e, est_loss)) 
        
        params = season_net.collect_params() # this returns a ParameterDict

print('The type of "params" is a ',type(params))

# A ParameterDict is a dictionary of Parameter class objects
# therefore, here is how we can read off the parameters from it.

for param in params.values():
    print(param.name,param.data())

# plot the convergence of the estimated loss function 
get_ipython().magic('matplotlib inline')

import matplotlib
import matplotlib.pyplot as plt

plt.figure(num=None,figsize=(8, 6),dpi=80, facecolor='w', edgecolor='k')
plt.semilogy(range(niter),loss_seq, '.')

# adding some additional bells and whistles to the plot
plt.grid(True,which="both")
plt.xlabel('iteration',fontsize=14)
plt.ylabel('est loss',fontsize=14)

fit = season_net(features[:train_length,:])
fct = season_net(features[train_length:,:])
mse = nd.mean(square_loss(fct,y[train_length:]))
print('Mean squared forecast error :%s' % mse)

def plot_forecast(observed, fitted, forecasted):
    plt.plot(fitted.asnumpy(), color="r")
    plt.plot(observed.asnumpy(), color="g")
    T = len(fitted)
    plt.plot(np.arange(T, T+len(forecasted)), forecasted.asnumpy(), color="b")
    plt.legend(["Fitted", "Observed", "Forecasted"]);
plot_forecast(y,fit,fct)

def gen_ar1(nobs, alpha, beta, sigma):
    y = nd.empty((nobs))
    ylag = nd.empty((nobs))
    epsilon = nd.random_normal(0, sigma, shape=(nobs)) # innovations
    y[0] = alpha + epsilon[0] #initial value
    ylag[0] = 0
    for t in range(nobs-1):
        y[t+1] = alpha + beta * y[t] + epsilon[t]
        ylag[t+1] = y[t]
    return y, ylag

y, ylag = gen_ar1(smpl_length,1,0.5,1)
plt.plot(y.asnumpy())

# We use the first 100 observations for training
ar_data = gluon.data.DataLoader(gluon.data.ArrayDataset(ylag[:train_length], y[:train_length]),
                                batch_size=batch_size, shuffle=True)

ar_ctx = mx.cpu()
ar_net = gluon.nn.Sequential()

with ar_net.name_scope():
    ar_net.add(gluon.nn.BatchNorm(axis=1, center=False, scale=True))
    ar_net.add(gluon.nn.Dense(1, in_units=1))
    
ar_net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=ar_ctx)

square_loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(ar_net.collect_params(), 'sgd', {'learning_rate': 0.01})


epochs = 30
smoothing_constant = .01
moving_loss = 0
niter = 0
loss_seq = []

for e in range(epochs):
    for i, (data, label) in enumerate(ar_data):
        data = data.as_in_context(ar_ctx)
        label = label.as_in_context(ar_ctx)
        with autograd.record():
            output = ar_net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        
        ##########################
        #  Keep a moving average of the losses
        ##########################
        niter +=1
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss

        # correct the bias from the moving averages
        est_loss = moving_loss/(1-(1-smoothing_constant)**niter)
        loss_seq.append(est_loss)
    if e % 10 ==0:
        print("Epoch %s. Moving avg of MSE: %s" % (e, est_loss)) 
        
        params = ar_net.collect_params() # this returns a ParameterDict

print('The type of "params" is a ',type(params))

# A ParameterDict is a dictionary of Parameter class objects
# therefore, here is how we can read off the parameters from it.

for param in params.values():
    print(param.name,param.data())

def gen_ar1_forecasts(last_obs,npred):
    fct = nd.zeros((npred,1))
    for i in range(npred):
        if i==0:
            fct[i,:] = ar_net(last_obs)
        else:
            ypred = nd.reshape(fct[i-1],shape=(1,1))
            fct[i,:] = ar_net(ypred)
    return fct
        
last_obs = nd.reshape(y[train_length-1],shape=(1,1))    
fct = gen_ar1_forecasts(last_obs,pred_length)

fit = ar_net(nd.reshape(ylag[:train_length],shape=(train_length,1)))
plot_forecast(y,fit,fct)

# estimate sigma_{ph}
residuals = y[:train_length]-fit
sigma_hat = np.std(residuals.asnumpy())
dense_weight = params["sequential1_dense0_weight"].data().asnumpy()
batchnorm_var = params["sequential1_batchnorm0_gamma"].data().asnumpy()
beta_hat = dense_weight*batchnorm_var

sigph = sigma_hat * np.sqrt(np.cumsum([pow(beta_hat,2*h) for h in range(pred_length)]))
sigph = np.reshape(sigph,(pred_length,1))
pred_interval = np.concatenate((fct.asnumpy() - 1.65*sigph,
                                fct.asnumpy() + 1.65*sigph),axis=1)

def plot_forecast_interval(observed, fitted, forecasted, interval):
    plt.plot(fitted.asnumpy(), color="r")
    plt.plot(observed.asnumpy(), color="g")
    T = len(fitted)
    plt.plot(np.arange(T, T+len(forecasted)), forecasted.asnumpy(), color="b")
    plt.fill_between(np.arange(T, T+len(forecasted)), interval[0,:], interval[1,:])
    plt.legend(["Fitted", "Observed", "Forecasted"]);

plot_forecast_interval(y,fit,fct,np.transpose(pred_interval))

def gen_ar1_prediction_interval(last_obs,npred,niter,resid):
    fct = nd.zeros((npred,niter))
    for i in range(niter):
        for j in range(npred):
            # resampling the residuals
            sample_indices = np.random.randint(resid.shape[0],size = npred)
            resamp = resid.asnumpy()[sample_indices,:]
            resamp = nd.array(resamp)
            if j==0:
                fct[j,i] = ar_net(last_obs) + resamp[j,:]
            else:
                ypred = nd.reshape(fct[j-1,i],shape=(1,1)) 
                fct[j,i] = ar_net(ypred) + resamp[j,:]
    # computing the prediction interval
    pred_interval = np.percentile(fct.asnumpy(),q=[5,95],axis=1)
    return pred_interval

# compute the residuals of the fitted model
resid = nd.reshape(y[:train_length], (train_length,1)) - fit
pred_interval = gen_ar1_prediction_interval(last_obs,pred_length,1000,resid)
plot_forecast_interval(y,fit,fct,pred_interval)

