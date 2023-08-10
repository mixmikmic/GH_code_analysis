import sys, os
import numpy
import time
sys.path.append(os.path.join(os.getcwd(),'..'))

import candlegp
from matplotlib import pyplot
import torch
from torch.autograd import Variable
get_ipython().run_line_magic('matplotlib', 'inline')
pyplot.style.use('ggplot')
import IPython

M = 50

def func(x):
    return torch.sin(x * 3*3.14) + 0.3*torch.cos(x * 9*3.14) + 0.5 * torch.sin(x * 7*3.14)
X = torch.rand(10000, 1).double() * 2 - 1
Y = func(X) + torch.randn(10000, 1).double() * 0.2
pyplot.plot(X.numpy(), Y.numpy(), 'x')
D = X.size(1)
Xt = torch.linspace(-1.1, 1.1, 100).double().unsqueeze(1)
Yt = func(Xt)

k = candlegp.kernels.RBF(D,variance=torch.DoubleTensor([1.0])).double()
Z = X[:M].clone()
m = candlegp.models.SVGP(Variable(X), Variable(Y.unsqueeze(1)),
                         likelihood=candlegp.likelihoods.Gaussian(ttype=torch.DoubleTensor),
                         kern=k, Z=Z)
m

# ground_truth = m.compute_log_likelihood() # seems to take too long
evals = []
for i in range(100):
    if i % 10 == 9:
        print ('.', end='')
    idxes = torch.randperm(X.size(0))[:100]
    evals.append(m.compute_log_likelihood(Variable(X[idxes]), Variable(Y[idxes])).data[0])

pyplot.hist(evals)
#pyplot.axvline(ground_truth)

mbps = numpy.logspace(-2, -0.8, 7)
times = []
objs = []
for mbp in mbps:
    minibatch_size = int(len(X) * mbp)
    print (minibatch_size)
    start_time = time.time()
    evals = []
    
    for i in range(20):
        idxes = torch.randperm(X.size(0))[:minibatch_size]
        evals.append(m.compute_log_likelihood(Variable(X[idxes]), Variable(Y[idxes])).data[0])
    objs.append(evals)

#    plt.hist(objs, bins = 100)
#    plt.axvline(ground_truth, color='r')
    times.append(time.time() - start_time)

f, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(16, 6))
ax1.plot(mbps, times, 'x-')
ax1.set_xlabel("Minibatch proportion")
ax1.set_ylabel("Time taken")

ax2.plot(mbps, numpy.array(objs), 'kx')
ax2.set_xlabel("Minibatch proportion")
ax2.set_ylabel("ELBO estimates")


pX = Variable(torch.linspace(-1, 1, 100).unsqueeze(1).double())
pY, pYv = m.predict_y(pX)
pyplot.plot(X.numpy(), Y.numpy(), 'x')
line, = pyplot.plot(pX.data.numpy(), pY.data.numpy(), lw=1.5)
col = line.get_color()
pyplot.plot(pX.data.numpy(), (pY+2*pYv**0.5).data.numpy(), col, lw=1.5)
pyplot.plot(pX.data.numpy(), (pY-2*pYv**0.5).data.numpy(), col, lw=1.5)
pyplot.plot(m.Z.get().data.numpy(), numpy.zeros(m.Z.shape), 'k|', mew=2)
pyplot.title("Predictions before training")



logt = []
logf = []

st = time.time()
minibatch_size = 100
m.Z.requires_grad = True
opt = torch.optim.Adam(m.parameters(), lr=0.01)
m.Z.requires_grad = False

for i in range(2000):
    if i % 50 == 49:
        print (i)
    idxes = torch.randperm(X.size(0))[:minibatch_size]
    opt.zero_grad()
    obj = m(Variable(X[idxes]), Variable(Y[idxes]))
    logf.append(obj.data[0])
    obj.backward()
    opt.step()
    logt.append(time.time() - st)
    if i%50 == 49:
        IPython.display.clear_output(True)
        pyplot.plot(-numpy.array(logf))
        pyplot.xlabel('iteration')
        pyplot.ylabel('ELBO')
        pyplot.show()



pX = Variable(torch.linspace(-1, 1, 100).unsqueeze(1).double())
pY, pYv = m.predict_y(pX)
pyplot.plot(X.numpy(), Y.numpy(), 'x')
line, = pyplot.plot(pX.data.numpy(), pY.data.numpy(), lw=1.5)
col = line.get_color()
pyplot.plot(pX.data.numpy(), (pY+2*pYv**0.5).data.numpy(), col, lw=1.5)
pyplot.plot(pX.data.numpy(), (pY-2*pYv**0.5).data.numpy(), col, lw=1.5)
pyplot.plot(m.Z.get().data.numpy(), numpy.zeros(m.Z.shape), 'k|', mew=2)
pyplot.title("Predictions after training")



