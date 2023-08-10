import sys, os
sys.path.append(os.path.join(os.getcwd(),'..'))

import candlegp
import candlegp.training.hmc
import numpy
import torch
from torch.autograd import Variable
from matplotlib import pyplot
pyplot.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


X = Variable(torch.linspace(-3,3,20,out=torch.DoubleTensor()))
Y = Variable(torch.from_numpy(numpy.random.exponential(((X.data.sin())**2).numpy())))

#build the model
k = candlegp.kernels.Matern32(1,ARD=False).double() + candlegp.kernels.Bias(1).double()
l = candlegp.likelihoods.Exponential()
m = candlegp.models.GPMC(X[:,None], Y[:,None], k, l)
m

m.kern.kern_list[0].lengthscales.prior = candlegp.priors.Gamma(1., 1., ttype=torch.DoubleTensor)
m.kern.kern_list[0].variance.prior = candlegp.priors.Gamma(1.,1., ttype=torch.DoubleTensor)
m.kern.kern_list[1].variance.prior = candlegp.priors.Gamma(1.,1., ttype=torch.DoubleTensor)
m.V.prior = candlegp.priors.Gaussian(0.,1., ttype=torch.DoubleTensor)
m

# start near MAP
opt = torch.optim.LBFGS(m.parameters(), lr=1e-2, max_iter=40)
def eval_model():
    obj = m()
    opt.zero_grad()
    obj.backward()
    return obj

for i in range(50):
    obj = m()
    opt.zero_grad()
    obj.backward()
    opt.step(eval_model)
    if i%5==0:
        print(i,':',obj.data[0])
m

res = candlegp.training.hmc.hmc_sample(m,500,0.2,burn=50, thin=10)

xtest = torch.linspace(-4,4,100).double().unsqueeze(1)

f_samples = []
for i in range(len(res[0])):
    for j,mp in enumerate(m.parameters()):
        mp.set(res[j+1][i])
    f_samples.append(m.predict_f_samples(Variable(xtest), 5).squeeze(0).t())
f_samples = torch.cat(f_samples, dim=0)

rate_samples = torch.exp(f_samples)
pyplot.figure(figsize=(12, 6))
line, = pyplot.plot(xtest.numpy(), rate_samples.data.mean(0).numpy(), lw=2)
pyplot.fill_between(xtest[:,0], numpy.percentile(rate_samples.data.numpy(), 5, axis=0), numpy.percentile(rate_samples.data.numpy(), 95, axis=0), color=line.get_color(), alpha = 0.2)
pyplot.plot(X.data.numpy(), Y.data.numpy(), 'kx', mew=2)
pyplot.ylim(-0.1, numpy.max(numpy.percentile(rate_samples.data.numpy(), 95, axis=0)))

import pandas
df = pandas.DataFrame(res[1:],index=[n for n,p in m.named_parameters()]).transpose()
df[:10]

df["kern.kern_list.1.variance"].apply(lambda x: x[0]).hist(bins=20)

Z = torch.linspace(-3,3,5).double().unsqueeze(1)
k2 = candlegp.kernels.Matern32(1,ARD=False).double() + candlegp.kernels.Bias(1).double()
l2 = candlegp.likelihoods.Exponential()
m2 = candlegp.models.SGPMC(X[:,None], Y[:,None], k2, l2, Z)
m2.kern.kern_list[0].lengthscales.prior = candlegp.priors.Gamma(1., 1., ttype=torch.DoubleTensor)
m2.kern.kern_list[0].variance.prior = candlegp.priors.Gamma(1.,1., ttype=torch.DoubleTensor)
m2.kern.kern_list[1].variance.prior = candlegp.priors.Gamma(1.,1., ttype=torch.DoubleTensor)
m2.V.prior = candlegp.priors.Gaussian(0.,1., ttype=torch.DoubleTensor)
m2

# start near MAP
opt = torch.optim.LBFGS(m2.parameters(), lr=1e-2, max_iter=40)
def eval_model():
    obj = m2()
    opt.zero_grad()
    obj.backward()
    return obj

for i in range(50):
    obj = m2()
    opt.zero_grad()
    obj.backward()
    opt.step(eval_model)
    if i%5==0:
        print(i,':',obj.data[0])
m2

res = candlegp.training.hmc.hmc_sample(m,500,0.2,burn=50, thin=10)

xtest = torch.linspace(-4,4,100).double().unsqueeze(1)

f_samples = []
for i in range(len(res[0])):
    for j,mp in enumerate(m.parameters()):
        mp.set(res[j+1][i])
    f_samples.append(m.predict_f_samples(Variable(xtest), 5).squeeze(0).t())
f_samples = torch.cat(f_samples, dim=0)

rate_samples = torch.exp(f_samples)
pyplot.figure(figsize=(12, 6))
line, = pyplot.plot(xtest.numpy(), rate_samples.data.mean(0).numpy(), lw=2)
pyplot.fill_between(xtest[:,0], numpy.percentile(rate_samples.data.numpy(), 5, axis=0), numpy.percentile(rate_samples.data.numpy(), 95, axis=0), color=line.get_color(), alpha = 0.2)
pyplot.plot(X.data.numpy(), Y.data.numpy(), 'kx', mew=2)
pyplot.plot(m2.Z.get().data.numpy(),numpy.zeros(m2.num_inducing),'o')
pyplot.ylim(-0.1, numpy.max(numpy.percentile(rate_samples.data.numpy(), 95, axis=0)))



