import h5py
import scipy
import numpy as np
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math
get_ipython().magic('matplotlib inline')

# generate some 1D regression data (reproducing Bishop book data, page 273). 
# Note that the P(y|x) is not a nice distribution. E.g. it has three modes for x ~= 0.5
N = 200
X = np.linspace(0,1,N)
Y = X + 0.3 * np.sin(2*3.1415926*X) + np.random.uniform(-0.1, 0.1, N)
X,Y = Y,X
plt.scatter(X,Y,color='g')

# model intialization
input_size = 1
hidden_size = 30
K = 3 # number of mixture components
m = {}
m['Wxh'] = np.random.randn(hidden_size, input_size) * 0.1 # input to hidden
m['Whu'] = np.random.randn(K, hidden_size) * 0.1 # hidden to means
m['Whs'] = np.random.randn(K, hidden_size) * 0.1 # hidden to log standard deviations
m['Whp'] = np.random.randn(K, hidden_size) * 0.1 # hidden to mixing coefficients (cluster priors)
m['bxh'] = np.random.randn(hidden_size, 1) * 0.01
m['bhu'] = np.random.randn(K, 1) * 0.01
m['bhs'] = np.random.randn(K, 1) * 0.01
m['bhp'] = np.random.randn(K, 1) * 0.01

def softmax(x):
    # softmaxes the columns of x
    #z = x - np.max(x, axis=0, keepdims=True) # for safety
    e = np.exp(x)
    en = e / np.sum(e, axis=0, keepdims=True)
    return en

def mdn_loss(x, y, m):
    # data in X are columns
    
    # forward pass
    h = np.tanh(np.dot(m['Wxh'], x) + m['bxh'])
    # predict mean
    mu = np.dot(m['Whu'], h) + m['bhu']
    # predict log variance
    logsig = np.dot(m['Whs'], h) + m['bhs']
    sig = np.exp(logsig)
    # predict mixture priors
    piu = np.dot(m['Whp'], h) + m['bhp'] # unnormalized pi
    pi = softmax(piu)
    # compute the loss: mean negative data log likelihood
    k,n = mu.shape # number of mixture components
    ps = np.exp(-((y - mu)**2)/(2*sig**2))/(sig*np.sqrt(2*math.pi))
    pin = ps * pi
    lp = -np.log(np.sum(pin, axis=0, keepdims=True))
    loss = np.sum(lp)/n
    
    # compute the gradients on nn outputs
    grad = {}
    gammas = pin / np.sum(pin, axis=0, keepdims = True)
    dmu = gammas * ((mu - y)/sig**2) /n
    dlogsig = gammas * (1.0 - (y-mu)**2/(sig**2)) /n
    dpiu = (pi - gammas) /n
    # backprop to decoder matrices
    grad['bhu'] = np.sum(dmu, axis=1, keepdims=True)
    grad['bhs'] = np.sum(dlogsig, axis=1, keepdims=True)
    grad['bhp'] = np.sum(dpiu, axis=1, keepdims=True)
    grad['Whu'] = np.dot(dmu, h.T)
    grad['Whs'] = np.dot(dlogsig, h.T)
    grad['Whp'] = np.dot(dpiu, h.T)
    # backprop to h
    dh = np.dot(m['Whu'].T, dmu) + np.dot(m['Whs'].T, dlogsig) + np.dot(m['Whp'].T, dpiu)
    # backprop tanh
    dh = (1.0-h**2)*dh
    # backprop input to hidden
    grad['bxh'] = np.sum(dh, axis=1, keepdims=True)
    grad['Wxh'] = np.dot(dh, x.T)
    
    # misc stats
    stats = {}
    stats['lp'] = lp
    return loss, grad, stats
    

# utility function for creating contour plot of the predictions
def drawContour(m):
    n = 50
    xx = np.linspace(0,1,n)
    yy = np.linspace(0,1,n)
    xm, ym = np.meshgrid(xx, yy)
    _, _, stats = mdn_loss(xm.reshape(1, xm.size), ym.reshape(1, ym.size), m)
    logps = stats['lp']
    plt.figure(figsize=(10,10))
    plt.scatter(X,Y,color='g')
    lp = stats['lp']
    plt.contour(xm, ym, np.reshape(logps, (n, n)), levels=np.linspace(lp.min(), lp.max(), 20))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('3-component Gaussian Mixture Model for P(y|x)')
#drawContour(m)

nb = N # full batch
xbatch = np.reshape(X[:nb], (1,nb))
ybatch = np.reshape(Y[:nb], (1,nb))

# optimize
lr = 1e-2
mem = {}
for k in m.keys(): mem[k] = np.zeros_like(m[k]) # init adagrad
for k in range(20000):
    loss, grad, stats = mdn_loss(xbatch, ybatch, m)
    if k % 1000 == 0:
        print '%d: %f' % (k, loss)
    for k,v in grad.iteritems():
        mem[k] += grad[k]**2
        m[k] += -lr * grad[k] / np.sqrt(mem[k] + 1e-8)
        

drawContour(m)

# Rung radient check

def wrapCost(batch, model):
    loss, grad, stats = mdn_loss(batch[0], batch[1], model)
    cg = {}
    cg['grad'] = grad
    cg['cost'] = {}
    cg['cost']['total_cost'] = loss
    return cg

def gradCheck(batch, model, cost_function, **kwargs):
    num_checks = kwargs.get('num_checks', 10)
    delta = kwargs.get('delta', 1e-5)
    rel_error_thr_warning = kwargs.get('rel_error_thr_warning', 1e-2)
    rel_error_thr_error = kwargs.get('rel_error_thr_error', 1)

    cg = cost_function(batch, model)

    print 'running gradient check...'
    for p in model.keys():
      print 'checking gradient on parameter %s of shape %s...' % (p, `model[p].shape`)
      mat = model[p]

      s0 = cg['grad'][p].shape
      s1 = mat.shape
      assert s0 == s1, 'Error dims dont match on %s: grad: %s vs. param: %s.' % (p, `s0`, `s1`)

      for i in xrange(num_checks):
        #ri = randi(mat.size)
        ri = np.random.randint(0, mat.size)

        # evluate cost at [x + delta] and [x - delta]
        old_val = mat.flat[ri]
        mat.flat[ri] = old_val + delta
        cg0 = cost_function(batch, model)
        mat.flat[ri] = old_val - delta
        cg1 = cost_function(batch, model)
        mat.flat[ri] = old_val # reset old value for this parameter

        # fetch both numerical and analytic gradient
        grad_analytic = cg['grad'][p].flat[ri]
        grad_numerical = (cg0['cost']['total_cost'] - cg1['cost']['total_cost']) / ( 2 * delta )

        # compare them
        if grad_numerical == 0 and grad_analytic == 0:
          rel_error = 0 # both are zero, OK.
          status = 'OK'
        elif abs(grad_numerical) < 1e-7 and abs(grad_analytic) < 1e-7:
          rel_error = 0 # not enough precision to check this
          status = 'VAL SMALL WARNING'
        else:
          rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
          status = 'OK'
          if rel_error > rel_error_thr_warning: status = 'WARNING'
          if rel_error > rel_error_thr_error: status = '!!!!! NOTOK'

        # print stats
        print '%s %s index %8d (val = %+8f), analytic = %+8f, numerical = %+8f, rel error = %+8f'               % (status, p, ri, old_val, grad_analytic, grad_numerical, rel_error)
            
gradCheck((xbatch, ybatch), m, wrapCost)



