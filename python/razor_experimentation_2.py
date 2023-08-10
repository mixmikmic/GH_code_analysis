import os
os.environ['THEANO_FLAGS'] = "device=cpu"

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import theano as th
import theano.tensor as T
import pymc3 as pm
import pickle
import csv
from statistics import median
from mpl_toolkits.mplot3d import axes3d
get_ipython().run_line_magic('matplotlib', 'notebook')

th.config.compute_test_value = 'ignore'

# Get the data
boxes = ['MultiJet', 'DiJet', 'LeptonMultiJet', 'LeptonJet']
sideband_x = {'MultiJet':650, 'DiJet':650, 'LeptonMultiJet':550, 'LeptonJet':550}
sideband_y = {'MultiJet':0.3, 'DiJet':0.3, 'LeptonMultiJet':0.2, 'LeptonJet':0.2}
btag_max = {'MultiJet':3, 'DiJet':2, 'LeptonMultiJet':3, 'LeptonJet':2}
data = { box:pd.read_csv("data/RazorFitInstance_Razor2016_MoriondRereco_{}.csv".format(box), index_col='NBtags')
           for box in boxes }
sideband_data = { box:data[box][ (data[box]['MR'] < sideband_x[box]) | (data[box]['Rsq'] < sideband_y[box]) ]
                    for box in boxes}

# Check the data
get_ipython().run_line_magic('matplotlib', 'notebook')
ax = sideband_data['LeptonMultiJet'].plot(kind='hexbin', x='MR', y='Rsq', bins='log', sharex=False);
plt.show()

class Sideband(object):
    """Class storing razor sideband geometry info."""
    def __init__(self, MRMin=500, MRMax=4000, RsqMin=0.25, RsqMax=1.5, 
                MRSidebandMax=650, RsqSidebandMax=0.30):
        self.MRMin = T.as_tensor_variable(MRMin)
        self.MRMax = T.as_tensor_variable(MRMax)
        self.RsqMin = T.as_tensor_variable(RsqMin)
        self.RsqMax = T.as_tensor_variable(RsqMax)
        self.MRSidebandMax = T.as_tensor_variable(MRSidebandMax)
        self.RsqSidebandMax = T.as_tensor_variable(RsqSidebandMax)
        
        # as floats
        self.MRMin_ = MRMin
        self.MRMax_ = MRMax
        self.RsqMin_ = RsqMin
        self.RsqMax_ = RsqMax
        self.MRSidebandMax_ = MRSidebandMax
        self.RsqSidebandMax_ = RsqSidebandMax
        
    def in_sideband(self, MR, Rsq):
        not_in = ( (MR < self.MRMin) + (MR > self.MRMax) + (Rsq < self.RsqMin) + (Rsq > self.RsqMax) 
                  + ((MR > self.MRSidebandMax)*(Rsq > self.RsqSidebandMax)) )
        return not_in < 1
        
hadronic_sideband = Sideband(500, 4000, 0.25, 1.5, 650, 0.30)
leptonic_sideband = Sideband(400, 4000, 0.15, 1.5, 550, 0.20)

hadronic_mr_sideband = Sideband(500, 650, 0.25, 1.5, 650, 1.5)
hadronic_rsq_sideband = Sideband(500, 4000, 0.25, 0.30, 4000, 0.30)

leptonic_mr_sideband = Sideband(400, 550, 0.15, 1.5, 550, 1.5)
leptonic_rsq_sideband = Sideband(400, 4000, 0.15, 0.20, 4000, 0.20)

# Using scipy's integration functionality we can directly compute the razor pdf integral numerically.
# This may allow us to move to pdfs whose norm is not analytically calculable.  

def razor_p(MR, Rsq, b, MR0):
    """Computes razor probability density (numpy version)."""
    term1 = b * (MR-MR0) * Rsq
    prob = np.exp(-term1)
    return prob

def razor_int(b, MR0, MRMin, MRMax, RsqMin, RsqMax):
    """Computes razor integral in a rectangular region"""
    func = lambda y, x: razor_p(x, y, b, MR0)
    RsqLow = lambda x: RsqMin
    RsqHigh = lambda x: RsqMax
    return sp.integrate.dblquad(func, MRMin, MRMax, RsqLow, RsqHigh)[0]

def razor_norm(b, MR0, sideband):
    norm_MR_sideband = razor_int(b, MR0, sideband.MRMin_, sideband.MRSidebandMax_, 
                                          sideband.RsqMin_, sideband.RsqMax_ )
    norm_Rsq_sideband = razor_int(b, MR0, sideband.MRSidebandMax_, sideband.MRMax_, 
                                          sideband.RsqMin_, sideband.RsqSidebandMax_ )
    return norm_MR_sideband + norm_Rsq_sideband

def razor_p_sidebandnormalized(MR, Rsq, b, MR0, nevents, sideband=hadronic_sideband):
    """Returns probability normalized to the given number of events in the given sideband"""
    p = razor_p(MR, Rsq, b, MR0)
    norm = razor_norm(b, MR0, sideband)
    return nevents*p/norm

class RazorIntegral(th.Op):
    """Inputs: b, MR0, MRMin, MRMax, RsqMin, RsqMax
       Output: pdf integral"""
    __props__ = ()
    epsilon = 1e-7
    
    def make_node(self, *inputs):
        inputs = [T.as_tensor_variable(inp) for inp in inputs]
        output = T.dscalar()
        return th.Apply(self, inputs, [output])
    
    def perform(self, node, inputs, output_storage):
        b, MR0, MRMin, MRMax, RsqMin, RsqMax = inputs
        func = lambda y, x: razor_p(x, y, b, MR0)
        RsqLow = lambda x: RsqMin
        RsqHigh = lambda x: RsqMax
        pdf_integral = sp.integrate.dblquad( func, MRMin, MRMax, RsqLow, RsqHigh )[0]
        
        res = output_storage[0]
        res[0] = np.array(pdf_integral)
        
    def grad(self, inputs, g):
        b, MR0, MRMin, MRMax, RsqMin, RsqMax = inputs
        out_grad = g[0]
        e = self.epsilon
        b_grad = (razor_integral(b + e, MR0, MRMin, MRMax, RsqMin, RsqMax) 
                  - razor_integral(b - e, MR0, MRMin, MRMax, RsqMin, RsqMax)) / (2 * e)
        MR0_grad = (razor_integral(b, MR0 + e, MRMin, MRMax, RsqMin, RsqMax) 
                  - razor_integral(b, MR0 - e, MRMin, MRMax, RsqMin, RsqMax)) / (2 * e)
        MRMin_grad = MRMin.zeros_like()
        MRMax_grad = MRMax.zeros_like()
        RsqMin_grad = RsqMin.zeros_like()
        RsqMax_grad = RsqMax.zeros_like()
        return [out_grad * gr for gr in [b_grad, MR0_grad, MRMin_grad, MRMax_grad, RsqMin_grad, RsqMax_grad]]
   
        
razor_integral = RazorIntegral()

# Create PyMC3 class for razor PDF
from pymc3.distributions import Continuous
from pymc3.distributions.dist_math import bound

class Razor(Continuous):
    def __init__(self, b, MR0, sideband, **kwargs):
        super(Razor, self).__init__(**kwargs)
        self.b = b
        self.MR0 = MR0
        self.sideband = sideband
        
    def in_sideband(self, MR, Rsq):
        return self.sideband.in_sideband(MR, Rsq)
        
    def norm(self):
        b = self.b
        MR0 = self.MR0
        norm_MR_sideband = razor_integral(b, MR0, self.sideband.MRMin, self.sideband.MRSidebandMax, 
                                          self.sideband.RsqMin, self.sideband.RsqMax )
        norm_Rsq_sideband = razor_integral(b, MR0, self.sideband.MRSidebandMax, self.sideband.MRMax, 
                                          self.sideband.RsqMin, self.sideband.RsqSidebandMax )
        return norm_MR_sideband + norm_Rsq_sideband

    def logp(self, values):
        MR = values[:,0]
        Rsq = values[:,1]
        b = self.b
        MR0 = self.MR0
        
        term1 = b * (MR-MR0) * Rsq
        logprob = -term1
        lognorm = T.log(self.norm())
        return bound(logprob - lognorm, b > 0, MR > MR0)

# Model setup and training

def build_model(data, nbtags, box='MultiJet',
               prior_slopes=[0.1, 0.001],
               initial_vals=[0.015, 500.], 
               sideband=None):
    """Create PyMC3 model with razor pdf and observed data.
        prior_slopes: slopes for the exponential prior on b, MR0
        initial_vals: test value for b, MR0"""
    with pm.Model() as model:
        # PDF variables
        b = pm.Exponential("b", prior_slopes[0], testval=initial_vals[0])
        pre_MR0 = pm.Exponential("pre_MR0", prior_slopes[1], testval=initial_vals[1])
        MR0 = pm.Deterministic("MR0", -pre_MR0 + 500)
        
        if sideband is not None:
            sband = sideband
        elif box in ['LeptonJet','LeptonMultiJet']:
            sband = leptonic_sideband
        else:
            sband = hadronic_sideband
            
        observed = data.ix[nbtags][['MR','Rsq']].values
        razor_pdf = Razor("razor_pdf", b=b, MR0=MR0, sideband=sband,
                          shape=len(observed), observed=observed)
    return model

def fit_model(model, samples=20000, burn_period=10000):
    """Fit the model using the NUTS sampler and return
        samples from the posterior"""
    with model:
        step = pm.Metropolis()
        trace = pm.sample(samples, step=step)
        burned_trace = trace[burn_period:]
    return burned_trace
    
def visualize_trace(trace):
    varnames = ['b', 'MR0']
    # Visualize posteriors and MCMC traces
    pm.plots.traceplot(trace=trace, varnames=varnames);
    plt.show()
    # Plot the autocorrelation of the MCMC traces.
    # Long tails on this plot mean that the Markov chain is not sampling uniformly from the posterior.
    pm.plots.autocorrplot(trace=trace, varnames=varnames);
    plt.show()
    # Visualize the posteriors for the function parameters
    pm.plot_posterior(trace=trace, varnames=varnames);
    plt.show()
    
def fit_and_plot(data, nbtags, samples=20000, burn_period=10000, sideband=None, box='MultiJet'):
    """Build the model, fit it, and make diagnostic plots.
        Returns the MCMC trace."""
    model = build_model(data, nbtags, sideband=sideband, box=box)
    trace = fit_model(model, samples, burn_period)
    visualize_trace(trace)
    return trace

def save_traces(traces, fname):
    pickle.dump(traces, open(fname, "wb"))
    
def load_traces(fname):
    return pickle.load(open(fname, "rb"))

def compare_fit_and_data(trace, d, sideband):
    xmin = sideband.MRMin_
    xmax = sideband.MRSidebandMax_
    ymin = sideband.RsqMin_
    ymax = sideband.RsqSidebandMax_
    
    xstep = 10.
    ystep = 0.01
    nx = int((xmax-xmin)/xstep)
    ny = int((ymax-ymin)/ystep)
    gridx_ = np.arange(xmin, xmax+xstep, xstep)
    gridy_ = np.arange(ymin, ymax+ystep, ystep)
    gridx, gridy = np.meshgrid(gridx_, gridy_)

    # Data
    binstepx = (xmax-xmin)/10.
    binstepy = (ymax-ymin)/10.
    data_bins = (np.arange(xmin, xmax+binstepx, binstepx), np.arange(ymin, ymax+binstepy, binstepy)) 
    data_hist = np.log(np.histogram2d(d["MR"], d["Rsq"], bins=data_bins)[0]).T - np.log(binstepx*binstepy)
    data_binsx, data_binsy = np.meshgrid((data_bins[0][:-1]+data_bins[0][1:])/2., 
                                         (data_bins[1][:-1]+data_bins[1][1:])/2.)
    data_norm = len(d)

    # Fit 
    pdf_vals = np.zeros((nx, ny))
    pdf_vals = np.log(razor_p_sidebandnormalized(gridx, gridy, median(trace["b"]), median(trace["MR0"]), 
                                             data_norm, sideband=sideband))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(gridx, gridy, pdf_vals)
    ax.plot_surface(data_binsx, data_binsy, data_hist)
    ax.set_xlabel('MR (GeV)')
    ax.set_ylabel('Rsq (GeV)')
    ax.set_zlabel('log density')
    plt.show()

# Visualize fit residuals (viziduals)
def vis_resids(trace, d, sideband):
    xmin = sideband.MRMin_
    xmax = min(2000, sideband.MRSidebandMax_)
    ymin = sideband.RsqMin_
    #ymax = min(1.0, sideband.RsqSidebandMax_)
    ymax = sideband.RsqSidebandMax_

    # Data
    nbinsx = 6.
    nbinsy = 6.
    if sideband.RsqSidebandMax_ > 1.:
        nbinsx = 6.
        nbinsy = 7.
    binstepx = (xmax-xmin)/nbinsx
    binstepy = (ymax-ymin)/nbinsy
    data_bins = [np.arange(xmin, xmax+binstepx, binstepx), np.arange(ymin, ymax+binstepy, binstepy)]
    if len(data_bins[1]) < nbinsy+1:
        data_bins[1].append(ymax)
    data_hist = np.histogram2d(d["MR"], d["Rsq"], bins=data_bins)[0]
    data_binsx, data_binsy = np.meshgrid((data_bins[0][:-1]+data_bins[0][1:])/2., 
                                         (data_bins[1][:-1]+data_bins[1][1:])/2., indexing='ij')
    data_norm = len(d[ (d['MR'] < xmax) & (d['Rsq'] < ymax) ])
    
    b = median(trace["b"])
    MR0 = median(trace["MR0"])
    pdf_vals = np.zeros((len(data_bins[0])-1, len(data_bins[1])-1))
    fit_norm = razor_norm(b, MR0, sideband)
    for bx,mr_low in enumerate(data_bins[0][:-1]):
        for by,rsq_low in enumerate(data_bins[1][:-1]):
            mr_high = data_bins[0][bx+1]
            rsq_high = data_bins[1][by+1]
            integral = razor_int(b, MR0, mr_low, mr_high, rsq_low, rsq_high) * data_norm / fit_norm
            pdf_vals[bx,by] = integral

    resids = np.clip(data_hist / pdf_vals, 0., 3.)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(data_binsx, data_binsy, resids)
    ax.set_xlabel('MR (GeV)')
    ax.set_ylabel('Rsq')
    ax.set_zlabel('Data/Fit')
    if ax.get_xlim()[1] > 2000:
        ax.set_xlim(xmin, 2000)
    ax.set_zbound(0., 3.)
    plt.show()

get_ipython().run_line_magic('matplotlib', 'inline')
traces = { box:[] for box in boxes }
for box in boxes:
    for btags in range(3):
        traces[box].append(fit_and_plot(sideband_data[box], btags, samples=4000, burn_period=2000, box=box))

for box in boxes:
    for btags in range(3):
        with open("traces_{}_{}b.csv".format(box, btags), "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['b','MR0'])
            for row in zip(traces[box][btags]['b'], traces[box][btags]['MR0']):
                writer.writerow(list(row))

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'MultiJet'
btags = 0
vis_resids(traces[box][btags], sideband_data[box].ix[btags], hadronic_mr_sideband)
vis_resids(traces[box][btags], sideband_data[box].ix[btags], hadronic_rsq_sideband)

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'MultiJet'
btags = 0
vis_resids(traces[box][btags], sideband_data[box].ix[btags], hadronic_mr_sideband)
vis_resids(traces[box][btags], sideband_data[box].ix[btags], hadronic_rsq_sideband)

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'MultiJet'
btags = 1
vis_resids(traces[box][btags], sideband_data[box].ix[btags], hadronic_mr_sideband)
vis_resids(traces[box][btags], sideband_data[box].ix[btags], hadronic_rsq_sideband)

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'MultiJet'
btags = 2
vis_resids(traces[box][btags], sideband_data[box].ix[btags], hadronic_mr_sideband)
vis_resids(traces[box][btags], sideband_data[box].ix[btags], hadronic_rsq_sideband)

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'DiJet'
btags = 0
vis_resids(traces[box][btags], sideband_data[box].ix[btags], hadronic_mr_sideband)
vis_resids(traces[box][btags], sideband_data[box].ix[btags], hadronic_rsq_sideband)

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'DiJet'
btags = 1
vis_resids(traces[box][btags], sideband_data[box].ix[btags], hadronic_mr_sideband)
vis_resids(traces[box][btags], sideband_data[box].ix[btags], hadronic_rsq_sideband)

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'DiJet'
btags = 2
vis_resids(traces[box][btags], sideband_data[box].ix[btags], hadronic_mr_sideband)
vis_resids(traces[box][btags], sideband_data[box].ix[btags], hadronic_rsq_sideband)

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'LeptonJet'
btags = 0
vis_resids(traces[box][btags], sideband_data[box].ix[btags], leptonic_mr_sideband)
vis_resids(traces[box][btags], sideband_data[box].ix[btags], leptonic_rsq_sideband)

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'LeptonJet'
btags = 1
vis_resids(traces[box][btags], sideband_data[box].ix[btags], leptonic_mr_sideband)
vis_resids(traces[box][btags], sideband_data[box].ix[btags], leptonic_rsq_sideband)

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'LeptonJet'
btags = 2
vis_resids(traces[box][btags], sideband_data[box].ix[btags], leptonic_mr_sideband)
vis_resids(traces[box][btags], sideband_data[box].ix[btags], leptonic_rsq_sideband)

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'LeptonMultiJet'
btags = 0
vis_resids(traces[box][btags], sideband_data[box].ix[btags], leptonic_mr_sideband)
vis_resids(traces[box][btags], sideband_data[box].ix[btags], leptonic_rsq_sideband)

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'LeptonMultiJet'
btags = 1
vis_resids(traces[box][btags], sideband_data[box].ix[btags], leptonic_mr_sideband)
vis_resids(traces[box][btags], sideband_data[box].ix[btags], leptonic_rsq_sideband)

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'LeptonMultiJet'
btags = 2
vis_resids(traces[box][btags], sideband_data[box].ix[btags], leptonic_mr_sideband)
vis_resids(traces[box][btags], sideband_data[box].ix[btags], leptonic_rsq_sideband)

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'MultiJet'
btags = 0
compare_fit_and_data(traces[box][btags], sideband_data[box].ix[btags], hadronic_mr_sideband)
compare_fit_and_data(traces[box][btags], sideband_data[box].ix[btags], hadronic_rsq_sideband)

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'MultiJet'
btags = 1
compare_fit_and_data(traces[box][btags], sideband_data[box].ix[btags], hadronic_mr_sideband)
compare_fit_and_data(traces[box][btags], sideband_data[box].ix[btags], hadronic_rsq_sideband)

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'MultiJet'
btags = 2
compare_fit_and_data(traces[box][btags], sideband_data[box].ix[btags], hadronic_mr_sideband)
compare_fit_and_data(traces[box][btags], sideband_data[box].ix[btags], hadronic_rsq_sideband)

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'DiJet'
btags = 0
compare_fit_and_data(traces[box][btags], sideband_data[box].ix[btags], hadronic_mr_sideband)
compare_fit_and_data(traces[box][btags], sideband_data[box].ix[btags], hadronic_rsq_sideband)

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'DiJet'
btags = 1
compare_fit_and_data(traces[box][btags], sideband_data[box].ix[btags], hadronic_mr_sideband)
compare_fit_and_data(traces[box][btags], sideband_data[box].ix[btags], hadronic_rsq_sideband)

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'DiJet'
btags = 2
compare_fit_and_data(traces[box][btags], sideband_data[box].ix[btags], hadronic_mr_sideband)
compare_fit_and_data(traces[box][btags], sideband_data[box].ix[btags], hadronic_rsq_sideband)

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'LeptonMultiJet'
btags = 0
compare_fit_and_data(traces[box][btags], sideband_data[box].ix[btags], leptonic_mr_sideband)
compare_fit_and_data(traces[box][btags], sideband_data[box].ix[btags], leptonic_rsq_sideband)

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'LeptonMultiJet'
btags = 1
compare_fit_and_data(traces[box][btags], sideband_data[box].ix[btags], leptonic_mr_sideband)
compare_fit_and_data(traces[box][btags], sideband_data[box].ix[btags], leptonic_rsq_sideband)

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'LeptonMultiJet'
btags = 2
compare_fit_and_data(traces[box][btags], sideband_data[box].ix[btags], leptonic_mr_sideband)
compare_fit_and_data(traces[box][btags], sideband_data[box].ix[btags], leptonic_rsq_sideband)

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'LeptonJet'
btags = 0
compare_fit_and_data(traces[box][btags], sideband_data[box].ix[btags], leptonic_mr_sideband)
compare_fit_and_data(traces[box][btags], sideband_data[box].ix[btags], leptonic_rsq_sideband)

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'LeptonJet'
btags = 1
compare_fit_and_data(traces[box][btags], sideband_data[box].ix[btags], leptonic_mr_sideband)
compare_fit_and_data(traces[box][btags], sideband_data[box].ix[btags], leptonic_rsq_sideband)

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'LeptonJet'
btags = 2
compare_fit_and_data(traces[box][btags], sideband_data[box].ix[btags], leptonic_mr_sideband)
compare_fit_and_data(traces[box][btags], sideband_data[box].ix[btags], leptonic_rsq_sideband)

# Visualize fit residuals (viziduals)
def vis_resids_1D(trace, d, sideband):
    xmin = sideband.MRMin_
    xmax = min(2000, sideband.MRSidebandMax_)
    ymin = sideband.RsqMin_
    ymax = sideband.RsqSidebandMax_

    # Data
    nbinsx = 6.
    nbinsy = 1.
    if sideband.RsqSidebandMax_ > 1.:
        nbinsx = 1
        nbinsy = 7.
    binstepx = (xmax-xmin)/nbinsx
    binstepy = (ymax-ymin)/nbinsy
    data_bins = [np.arange(xmin, xmax+binstepx, binstepx), np.arange(ymin, ymax+binstepy, binstepy)]
    if len(data_bins[1]) < nbinsy+1:
        data_bins[1].append(ymax)
    data_hist = np.histogram2d(d["MR"], d["Rsq"], bins=data_bins)[0]
    data_binsx, data_binsy = np.meshgrid((data_bins[0][:-1]+data_bins[0][1:])/2., 
                                         (data_bins[1][:-1]+data_bins[1][1:])/2., indexing='ij')
    data_norm = len(d[ (d['MR'] < xmax) & (d['Rsq'] < ymax) ])
    
    b = median(trace["b"])
    MR0 = median(trace["MR0"])
    pdf_vals = np.zeros((len(data_bins[0])-1, len(data_bins[1])-1))
    fit_norm = razor_norm(b, MR0, sideband)
    for bx,mr_low in enumerate(data_bins[0][:-1]):
        for by,rsq_low in enumerate(data_bins[1][:-1]):
            mr_high = data_bins[0][bx+1]
            rsq_high = data_bins[1][by+1]
            integral = razor_int(b, MR0, mr_low, mr_high, rsq_low, rsq_high) * data_norm / fit_norm
            pdf_vals[bx,by] = integral

    resids = np.squeeze(np.clip(data_hist / pdf_vals, 0., 3.))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if sideband.RsqSidebandMax_ > 1.:
        bins = np.squeeze(data_binsy)
        ax.set_xlabel('Rsq')
    else:
        bins = np.squeeze(data_binsx)
        ax.set_xlabel('MR (GeV)')
    ax.plot(bins, resids)
    ax.set_ylabel('Data/Fit')
    if ax.get_xlim()[1] > 2000:
        ax.set_xlim(xmin, 2000)
    plt.show()

get_ipython().run_line_magic('matplotlib', 'notebook')
box = 'MultiJet'
btags = 0
vis_resids_1D(traces[box][btags], sideband_data[box].ix[btags], hadronic_mr_sideband)
vis_resids_1D(traces[box][btags], sideband_data[box].ix[btags], hadronic_rsq_sideband)

# Get nsigma values in every sideband bin

bins_hadronic = ([500, 575, 650, 750, 900, 1200, 1600, 2500, 4000], [0.25,0.30,0.41,0.52,0.64,1.5])
bins_leptonic = ([400, 475, 550, 700, 900, 1200, 1600, 2500, 4000], [0.15,0.20,0.25,0.30,0.41,0.52,0.64,1.5])

def get_data_yield(d, bin_coords):
    mrmin, mrmax, rsqmin, rsqmax = bin_coords
    return len(d[ (d['MR'] > mrmin) & (d['MR'] < mrmax) & (d['Rsq'] > rsqmin) & (d['Rsq'] < rsqmax) ])

def get_pred_yields(trace, bin_coords, sband, norm):
    mrmin, mrmax, rsqmin, rsqmax = bin_coords
    yields = []
    for i in range(len(trace)):
        b = trace['b'][i]
        MR0 = trace['MR0'][i]
        full_int = razor_norm(b, MR0, sband)
        yields.append( razor_int(b, MR0, mrmin, mrmax, rsqmin, rsqmax)*norm/full_int )
    return np.array(yields)

def get_nsigma(obs, preds):
    central_pred = median(preds)
    total = float(len(preds))
    if obs > central_pred:
        cdf = 1 - len(preds[preds > obs])/total
    else:
        cdf = len(preds[preds < obs])/total
    print obs, central_pred, sp.stats.iqr(preds), cdf
    return sp.stats.norm.ppf(cdf)

def get_nsigmas(trace, d, box):
    bins = bins_hadronic
    sband = hadronic_sideband
    if 'Lepton' in box:
        bins = bins_leptonic
        sband = leptonic_sideband
    norm = len(d)
    nsigmas = {}
    for ix, mrlow in enumerate(bins[0][:-1]):
        mrhigh = bins[0][ix+1]
        # only consider bins in sideband (2 bins in MR, 1 in Rsq)
        if ix < 2:
            iymax = len(bins[1])-1
        else:
            iymax = 1
        for iy, rsqlow in enumerate(bins[1][:iymax]):
            rsqhigh = bins[1][iy+1]
            this_bin = (mrlow, mrhigh, rsqlow, rsqhigh)
            this_yield = get_data_yield(d, this_bin)
            this_preds = get_pred_yields(trace, this_bin, sband, norm)
            nsigmas[this_bin] = get_nsigma(this_yield, this_preds)
    return nsigmas

get_nsigmas(traces['LeptonMultiJet'][0], sideband_data["LeptonMultiJet"].ix[0], 'LeptonMultiJet')



