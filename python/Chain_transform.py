get_ipython().run_line_magic('pylab', 'inline')

import numpy as np
import pymc3 as pm
import pymc3.distributions.transforms as tr
import theano.tensor as tt
import theano
import numpy.testing as npt

from pymc3.distributions.transforms import Transform

Order = tr.Ordered()
upper, lower = 0., 1.
Interval = tr.Interval(upper, lower)
Logodd = tr.LogOdds()
chain_tran = tr.Chain([Logodd, Order])

with pm.Model() as m0:
    x = pm.Uniform('x', 0., 1., shape=2,
                   transform=chain_tran,
                   testval=[0.1, 0.9])
    pm.Deterministic('jacobian', chain_tran.jacobian_det(chain_tran.forward(x)))
    
    tr0 = pm.sample(5000, tune=1000)

varnames = tr0.varnames
pm.traceplot(tr0, varnames=['jacobian']);

_, ax = plt.subplots(1, 2, figsize=(10, 5))
for ivar, varname in enumerate(varnames[:2]):
    ax[ivar].scatter(tr0[varname][:, 0], tr0[varname][:, 1], alpha=.01)
    ax[ivar].set_title(varname)
plt.tight_layout();

shape = (4, 2)
testval = np.random.rand(*shape)
testval = np.sort(testval/testval.sum(axis=-1, keepdims=True))
testval

with pm.Model() as m1:
    x = pm.Normal('x', 0., 1., shape=shape,
                   transform=Order,
                   testval=testval)
    
    tr1 = pm.sample(5000, tune=1000)

factors = [var.logpt for var in m1.basic_RVs] + m1.potentials
func1 = theano.function(m1.basic_RVs, factors)
func1(Order.forward_val(testval))

m1.logp(m1.test_point)

x0 = m1.basic_RVs[0]
x0.distribution.logp(Order.forward_val(testval)).eval()

x = Order.forward_val(testval)
x0.distribution.logp_nojac(x).eval()

x0.distribution.transform_used.jacobian_det(x).eval()

(x0.distribution.logp_nojac(x).sum(axis=-1) + x0.distribution.transform_used.jacobian_det(x)).eval()

tt.sum(x0.distribution.logp_nojac(x).sum(axis=-1) + x0.distribution.transform_used.jacobian_det(x)).eval()

with pm.Model() as m2:
    x = pm.Uniform('x', 0., 1., shape=shape,
                   transform=chain_tran,
                   testval=testval)
    pm.Deterministic('jacobian', chain_tran.jacobian_det(chain_tran.forward(x)))
    
    tr2 = pm.sample(2500, tune=1000)

post_x = tr2['x']

_, ax = plt.subplots(1, 4, figsize=(12, 3))
for ivar in range(shape[0]):
    ax[ivar].scatter(post_x[:, ivar, 0], post_x[:, ivar, 1], alpha=.01)
plt.tight_layout();

print(x.ndim, m2.free_RVs[0].logp_elemwiset.ndim)

x_ = tt.sum(x[..., 1:], axis=-1)
x_.ndim

def check_elementwise_logp_vector_transform(model, opt=0):
    x0 = model.deterministics[0]
    x = model.free_RVs[0]
    npt.assert_equal(x.ndim-1, x.logp_elemwiset.ndim)
    
    pt = model.test_point
    array = np.random.randn(*model.bijection.map(pt).shape)
    pt2 = model.bijection.rmap(array)
    dist = x.distribution
    logp_nojac = x0.distribution.logp(dist.transform_used.backward(pt2[x.name]))
    jacob_det = dist.transform_used.jacobian_det(pt2[x.name])
    npt.assert_equal(x.logp_elemwiset.ndim, jacob_det.ndim)
    
    if opt==0:
        elementwiselogp = logp_nojac.sum(axis=-1) + jacob_det
    else:
        elementwiselogp = logp_nojac + jacob_det
                      
    npt.assert_array_almost_equal(x.logp_elemwise(pt2),
                                  elementwiselogp.eval())
    
check_elementwise_logp_vector_transform(m0)
check_elementwise_logp_vector_transform(m1)
check_elementwise_logp_vector_transform(m2)

with pm.Model() as m:
    x = pm.Normal('x', shape=(4, 2))
print(x.ndim, m.free_RVs[0].logp_elemwiset.ndim)

x_ = tt.sum(x[..., 1:], axis=-1)
x_.ndim

with pm.Model() as m:
    x = pm.Normal('x', shape=2)
print(x.ndim, m.free_RVs[0].logp_elemwiset.ndim)

x_ = tt.sum(x[..., 1:], axis=-1)
x_.ndim

with pm.Model() as m:
    x = pm.Dirichlet('x', a=np.ones(3))

check_elementwise_logp_vector_transform(m, opt=1)

with pm.Model() as m:
    x = pm.Dirichlet('x', a=np.ones((2, 3)), shape=(2, 3))

check_elementwise_logp_vector_transform(m, opt=1)

with pm.Model() as m:
    x = pm.Uniform('x', shape=3, testval=np.ones(3),
                     transform=tr.stick_breaking)

check_elementwise_logp_vector_transform(m, opt=0)

with pm.Model() as m:
    x = pm.Uniform('x', shape=(4, 3), testval=np.ones((4, 3)),
                     transform=tr.stick_breaking)

check_elementwise_logp_vector_transform(m, opt=0)

with pm.Model() as m:
    x = pm.Normal('x', shape=3, testval=np.ones(3)/3,
                     transform=tr.sum_to_1)

check_elementwise_logp_vector_transform(m, opt=0)

with pm.Model() as m:
    x = pm.Normal('x', shape=(4, 3), testval=np.ones((4, 3))/3,
                     transform=tr.sum_to_1)

check_elementwise_logp_vector_transform(m, opt=0)

with pm.Model() as m:
    x = pm.MvNormal('x',
                    mu=np.zeros(3),
                    cov=np.diag(np.ones(3)),
                    shape=(4, 3), 
                    testval=np.sort(np.random.randn(4, 3)),
                    transform=tr.ordered)

check_elementwise_logp_vector_transform(m, opt=1)

def check_elementwise_logp_transform(model):
    x0 = model.deterministics[0]
    x = model.free_RVs[0]
    npt.assert_equal(x.ndim, x.logp_elemwiset.ndim)
    
    pt = model.test_point
    array = np.random.randn(*model.bijection.map(pt).shape)
    pt2 = model.bijection.rmap(array)
    dist = x.distribution
    logp_nojac = x0.distribution.logp(dist.transform_used.backward(pt2[x.name]))
    jacob_det = dist.transform_used.jacobian_det(pt2[x.name])
    npt.assert_equal(x.logp_elemwiset.ndim, jacob_det.ndim)
    
    elementwiselogp = logp_nojac + jacob_det
                      
    npt.assert_array_almost_equal(x.logp_elemwise(pt2),
                                  elementwiselogp.eval())

with pm.Model() as m:
    x = pm.VonMises('x', -2.5, 2., shape=(5, 3))

check_elementwise_logp_transform(m)

with pm.Model() as m:
    x = pm.VonMises('x', -2.5, 2.)
    trace = pm.sample(5000, tune=1000)

pm.traceplot(trace, priors=[x.distribution]);

Order = tr.Ordered()
Logodd = tr.LogOdds()
stickbreak = tr.StickBreaking()
sumto1 = tr.SumTo1()

chain_tran = tr.Chain([sumto1, Logodd])

x = np.random.randn(4, 2)
y = chain_tran.backward(x).eval()
y

chain_tran.forward(y).eval()

x

with pm.Model() as m:
    x = pm.Uniform('x', 0., 1., shape=shape,
                   transform=chain_tran,
                   testval=testval)

check_elementwise_logp_vector_transform(m, opt=0)

x_tr = m.free_RVs[0]
jac = x_tr.distribution.transform_used.jacobian_det(testval)
print(x.ndim, x_tr.logp_elemwiset.ndim, jac.ndim)

with m:
    tr1 = pm.sample()

post_x = tr1['x']
_, ax = plt.subplots(1, 4, figsize=(12, 3))
for ivar in range(shape[0]):
    ax[ivar].scatter(post_x[:, ivar, 0], post_x[:, ivar, 1], alpha=.01)
plt.tight_layout();

x_tr.distribution.logp_nojac(m.test_point[x_tr.name]).eval()

x_tr.distribution.transform_used.jacobian_det(m.test_point[x_tr.name]).eval()

trlist = x_tr.distribution.transform_used.transform_list

trlist[-1].jacobian_det(theano.shared(m.test_point[x_tr.name])).eval()

(trlist[0]
 .jacobian_det(
    trlist[1].backward(
        theano.shared(m.test_point[x_tr.name])
    )
 )
).eval()

chain_tran = tr.Chain([sumto1, Logodd, Order])

x = np.random.randn(4, 2)
y = chain_tran.backward(x).eval()
y

chain_tran.forward(y).eval()

x

with pm.Model() as m:
    x = pm.Uniform('x', 0., 1., shape=shape,
                   transform=chain_tran,
                   testval=testval)

check_elementwise_logp_vector_transform(m, opt=0)

with m:
    tr1 = pm.sample()

post_x = tr1['x']
_, ax = plt.subplots(1, 4, figsize=(12, 3))
for ivar in range(shape[0]):
    ax[ivar].scatter(post_x[:, ivar, 0], post_x[:, ivar, 1], alpha=.01)
plt.tight_layout();

chain_tran = tr.Chain([sumto1, Interval])
with pm.Model() as m:
    x = pm.Uniform('x', 
                     shape=shape, 
                     transform=chain_tran, 
                     testval=testval)
    x_ = pm.Deterministic('x_', tt.sort(x))

check_elementwise_logp_vector_transform(m, opt=0)

with m:
    tr1 = pm.sample()

post_x = tr1['x_']
_, ax = plt.subplots(1, 4, figsize=(12, 3))
for ivar in range(shape[0]):
    ax[ivar].scatter(post_x[:, ivar, 0], post_x[:, ivar, 1], alpha=.01)
plt.tight_layout();



