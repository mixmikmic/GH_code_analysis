import math
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (9,6)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
plt.rc('pdf', fonttype=42)
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True
plt.rcParams['font.serif'] = 'Times'
plt.rcParams['font.family'] = 'serif'

def linear(x):
    return x

def relu(x):
    return(max(0,x))

def sigmoid(x):
    return 1/(1 + math.exp(-x)) 

def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

vals = np.arange(-10.0, 10.0, 0.1)

l = [linear(x) for x in vals]
r = [relu(x) for x in vals]
s = [sigmoid(x) for x in vals]
t = [tanh(x) for x in vals]
transformations = [l,r,s,t]
titles = ['Linear', "ReLU", "Sigmoid", "Hyperbolic Tangent"]

fig = plt.figure()
for i, label in enumerate(('$f(x) = x$', 
                           '$f(x) = max(0,x)$', 
                           '$f(x) = \\frac{1}{1 + e^-x}$', 
                           '$f(x) = \\frac{e^x - e^-x}{e^x + e^-x}$')):
    ax = fig.add_subplot(2,2,i+1)
    ax.plot(l, transformations[i])
    ax.text(0.05, 0.95, label, transform=ax.transAxes,
      fontsize=14, va='top')
    ax.set_ylabel('$f(x)$',fontsize=12)
    ax.set_xlabel('$x$',fontsize=12)
    ax.set_title(titles[i])
fig.subplots_adjust(hspace=.4, wspace=.3)
plt.savefig('activations.pdf')
plt.show()

