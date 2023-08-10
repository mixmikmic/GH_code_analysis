import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
get_ipython().magic('matplotlib inline')

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

xs = Variable(torch.linspace(-10, 10, 5001))

xs.size()

ys = Variable(torch.ones(1))

def N(μ, σ2):
    def gaussian(x):
        nonlocal μ, σ2
        z = 2 * np.pi * σ2
        return 1/torch.sqrt(z.expand_as(x)) * torch.exp(-(x - μ.expand_as(x))**2 / z.expand_as(x))
    
    return gaussian

μ = Variable(torch.ones(1)*1, requires_grad=True)
σ2 = Variable(torch.ones(1), requires_grad=True)

Q = N(μ, σ2)

ys = Q(xs)

plt.figure(figsize=(4,2.5), dpi=110)
plt.title('Gaussian distribution')
plt.axvline(x=1, color='grey')
plt.ylim(-0.0625, 0.5)
plt.plot(xs.data.numpy(), ys.data.numpy(), linewidth=5, color='red', alpha=0.3);

P = lambda x: 0.75 * N(Variable(torch.ones(1) * -0.125), Variable(torch.ones(1)*0.01))(x) +     0.25 * N(Variable(torch.ones(1)), Variable(torch.ones(1)*0.01))(x)

plt.figure(figsize=(4, 2.5), dpi=110)
plt.title('Data Distribution (Normalized)')
plt.plot(xs.data.numpy(), P(xs).data.numpy(), alpha=0.5, color='#23aaff', linewidth=4)
plt.xlim(-3, 3);

ps = P(xs)

qs = Q(xs)

torch.sum(torch.log(ps/qs))

P = lambda x: 0.75 * N(Variable(torch.zeros(1)), Variable(torch.ones(1)*0.01))(x) +     0.25 * N(Variable(torch.ones(1)), Variable(torch.ones(1)*0.01))(x)

get_ipython().magic('matplotlib notebook')
time.sleep(3)

fig = plt.figure(figsize=(10, 3), dpi=70);

μ = Variable(torch.ones(1)*3, requires_grad=True)
σ2 = Variable(torch.ones(1), requires_grad=True)

P = N(Variable(torch.ones(1)*-1.5), Variable(torch.ones(1) * 2))
Q = N(μ, σ2)

α = 1e-12
steps = int(7e2)

plt.plot(xs.data.numpy(), P(xs).data.numpy(), alpha=0.5, color='red', linewidth=8)

optimizer = optim.SGD([μ, σ2], lr=0.0005)
for i in range(steps):
    optimizer.zero_grad()
    ps = P(xs) + α
    qs = Q(xs) + α
    loss = torch.sum(torch.log(ps/qs) * ps)
    loss.backward()
    optimizer.step()
    
    fig.canvas.draw()
    if i % 25 == 0:
        plt.plot(xs.data.numpy(), qs.data.numpy())
        time.sleep(0.1)
    if i % 50 == 0:    
        print(loss.data.numpy()[0], end=', ')

plt.figure(figsize=(10, 3), dpi=70)
plt.title("Showing Fit Result")
plt.plot(xs.data.numpy(), P(xs).data.numpy(), alpha=0.5, color='red', linewidth=8, label='P')
plt.plot(xs.data.numpy(), qs.data.numpy(), label='fit')
plt.ylim(-0.0625, 0.45)
plt.legend(framealpha=1, edgecolor='none')

fig = plt.figure(figsize=(10, 3), dpi=70)

μ = Variable(torch.ones(1), requires_grad=True)
σ2 = Variable(torch.ones(1), requires_grad=True)

P = lambda x: 0.75 * N(Variable(torch.zeros(1)), Variable(torch.ones(1)*0.05))(x) +     0.25 * N(Variable(1.5 * torch.ones(1)), Variable(torch.ones(1)*0.05))(x)
Q = N(μ, σ2)

α = 1e-12
steps = int(1e2)

plt.plot(xs.data.numpy(), P(xs).data.numpy(), alpha=0.5, color='red', linewidth=4)

optimizer = optim.Adam([μ, σ2], lr=0.05)
for i in range(steps):
    optimizer.zero_grad()
    ps = P(xs) + α
    qs = Q(xs) + α
    loss = torch.sum(torch.log(ps/qs) * ps)
    loss.backward()
    optimizer.step()
    
    fig.canvas.draw()
    if i % 5 == 0:
        plt.plot(xs.data.numpy(), qs.data.numpy())
        plt.xlim(-7.5, 7.5)
        time.sleep(0.1)
    if i % 20 == 0:    
        print(loss.data.numpy()[0], end=', ')

qs_klpq = qs

plt.figure(figsize=(10, 3), dpi=70)
plt.title("Showing Fit Result")
plt.plot(xs.data.numpy(), ps.data.numpy(), alpha=0.5, color='red', linewidth=4, label='P')
plt.plot(xs.data.numpy(), qs.data.numpy(), label='fit')
plt.xlim(-7.5, 7.5)
plt.legend(framealpha=1, edgecolor='none');

fig = plt.figure(figsize=(10, 3), dpi=70)

μ = Variable(torch.ones(1), requires_grad=True)
σ2 = Variable(torch.ones(1), requires_grad=True)

P = lambda x: 0.75 * N(Variable(-2 * torch.zeros(1)), Variable(torch.ones(1)*0.05))(x) +     0.25 * N(Variable(1.5 * torch.ones(1)), Variable(torch.ones(1)*0.05))(x)
Q = N(μ, σ2)

α = 1e-12
steps = int(3e2)

plt.plot(xs.data.numpy(), P(xs).data.numpy(), alpha=0.5, color='red', linewidth=4)

optimizer = optim.SGD([μ, σ2], lr=0.0001)
for i in range(steps):
    optimizer.zero_grad()
    ps = P(xs) + α
    qs = Q(xs) + α
    loss = torch.sum(torch.log(qs/ps) * qs)
    loss.backward()
    optimizer.step()
    
    fig.canvas.draw()
    if i % 5 == 0:
        plt.plot(xs.data.numpy(), qs.data.numpy());
        plt.xlim(-7.5, 7.5)
        time.sleep(0.1)
    if i % 20 == 0:    
        print(loss.data.numpy()[0], end=', ')

qs_klqp = qs

plt.figure(figsize=(10, 3), dpi=70)
plt.title("Showing Fit Result")
plt.plot(xs.data.numpy(), P(xs).data.numpy(), alpha=0.5, color='red', linewidth=4, label='P')
plt.plot(xs.data.numpy(), qs.data.numpy(), label='fit KL(P || Q)')
plt.xlim(-7.5, 7.5)
plt.legend(framealpha=1, edgecolor='none');

plt.figure(figsize=(10, 3), dpi=70)
plt.title("KL(P||Q) vs KL(Q||P) with Small Inter-modal Separation", fontsize=17)
plt.plot(xs.data.numpy(), P(xs).data.numpy(), alpha=0.5, color='red', linewidth=4, label='P')
plt.plot(xs.data.numpy(), qs_klpq.data.numpy(), label='Q, fit KL(P || Q)', color='#2a0dce')
plt.plot(xs.data.numpy(), qs_klqp.data.numpy(), label='Q, fit KL(Q || P)', color='#0ece84')
plt.xlim(-7.5, 7.5)
plt.legend(framealpha=1, edgecolor='none', fontsize=15)
plt.savefig('Comparing KL P Q vs KL Q P with small separation.png')

fig = plt.figure(figsize=(10, 3), dpi=70)
plt.title('KL(Q || P) for larger separation', fontsize=17);

μ = Variable(torch.ones(1), requires_grad=True)
σ2 = Variable(torch.ones(1), requires_grad=True)

P = lambda x: 0.75 * N(Variable(torch.ones(1) * -1), Variable(torch.ones(1)*0.05))(x) +     0.25 * N(Variable(1.5 * torch.ones(1)), Variable(torch.ones(1)*0.05))(x)
Q = N(μ, σ2)

α = 1e-12
steps = int(3e2)

plt.plot(xs.data.numpy(), P(xs).data.numpy(), alpha=0.5, color='red', linewidth=4)

optimizer = optim.Adam([μ, σ2], lr=0.05)
for i in range(steps):
    optimizer.zero_grad()
    ps = P(xs) + α
    qs = Q(xs) + α
    loss = torch.sum(torch.log(qs/ps) * qs)
    loss.backward()
    optimizer.step()
    
    fig.canvas.draw()
    if i % 5 == 0:
        plt.plot(xs.data.numpy(), qs.data.numpy());
        plt.xlim(-7.5, 7.5)
        time.sleep(0.1)
    if i % 20 == 0:    
        print(loss.data.numpy()[0], end=', ')

qs_klqp_ls = qs

plt.figure(figsize=(10, 3), dpi=70)
plt.title("Showing Fit Result", fontsize=17)
plt.plot(xs.data.numpy(), P(xs).data.numpy(), alpha=0.5, color='red', linewidth=4, label='P')
plt.plot(xs.data.numpy(), qs.data.numpy(), label='fit KL(Q || P)')
plt.legend(framealpha=1, edgecolor='none', fontsize=15);

fig = plt.figure(figsize=(10, 3), dpi=70)
plt.title('KL(P || Q) for larger separation', fontsize=17);

μ = Variable(torch.ones(1), requires_grad=True)
σ2 = Variable(torch.ones(1), requires_grad=True)

P = lambda x: 0.75 * N(Variable(torch.ones(1) * -1), Variable(torch.ones(1)*0.05))(x) +     0.25 * N(Variable(1.5 * torch.ones(1)), Variable(torch.ones(1)*0.05))(x)
Q = N(μ, σ2)

α = 1e-12
steps = int(3e2)

plt.plot(xs.data.numpy(), P(xs).data.numpy(), alpha=0.5, color='red', linewidth=4)

optimizer = optim.Adam([μ, σ2], lr=0.05)
for i in range(steps):
    optimizer.zero_grad()
    ps = P(xs) + α
    qs = Q(xs) + α
    loss = torch.sum(torch.log(ps/qs) * ps)
    loss.backward()
    optimizer.step()
    
    fig.canvas.draw()
    if i % 5 == 0:
        plt.plot(xs.data.numpy(), qs.data.numpy());
        plt.xlim(-7.5, 7.5)
        time.sleep(0.1)
    if i % 20 == 0:    
        print(loss.data.numpy()[0], end=', ')

qs_klpq_ls = qs

plt.figure(figsize=(10, 3), dpi=70)
plt.title("KL(P||Q) vs KL(Q||P) with large inter-modal separation", fontsize=17)
plt.plot(xs.data.numpy(), P(xs).data.numpy(), alpha=0.5, color='red', linewidth=4, label='P')
plt.plot(xs.data.numpy(), qs_klpq_ls.data.numpy(), label='Q, fit KL(P || Q)', color='#2a0dce')
plt.plot(xs.data.numpy(), qs_klqp_ls.data.numpy(), label='Q, fit KL(Q || P)', color='#0ece84')
plt.xlim(-7.5, 7.5)
plt.legend(framealpha=1, edgecolor='none', fontsize=15)
plt.savefig('Comparing KL P Q vs KL Q P with large separation.png')



