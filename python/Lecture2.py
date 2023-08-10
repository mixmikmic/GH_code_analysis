from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
from IPython.display import HTML, IFrame, Image, SVG, Latex
import ROOT
from ROOT import RooFit, RooStats
get_ipython().magic('matplotlib inline')
#%matplotlib nbagg
#%matplotlib notebook
from ipywidgets import interact, interactive, fixed
import colormaps

from matplotlib import rcParams
rcParams.update({'font.size': 18})
HTML('<link rel="stylesheet" href="custom.css" type="text/css">')

#from notebook.services.config import ConfigManager
#cm = ConfigManager()
#cm.update('livereveal', {
#          'theme': 'sans',
#          'transition': 'zoom',
#})

def iter_collection(rooAbsCollection):
    iterator = rooAbsCollection.createIterator()
    object = iterator.Next()
    while object:
        yield object
        object = iterator.Next()

def RooDataSet2pandas(data):
    nevents = data.numEntries()
    columns = [x.GetName() for x in iter_collection(data.get(0))]
    return pd.DataFrame([[x.getVal() for x in iter_collection(data.get(ievent))] for ievent in xrange(nevents)], columns=columns)

def p2z(p_value, onetail=True):
    """ pvalue to significance """
    if not onetail:
        p_value /= 2
    return stats.norm.isf(p_value)  # inverse of the survival function

def z2p(z, onetail=True):
    """ significance to pvalue """
    if onetail:
        return stats.norm.sf(z)
    else:
        return stats.norm.sf(z) * 2

Image("img/type-i-and-type-ii-errors.jpg", width="70%")  # from https://effectsizefaq.files.wordpress.com/2010/05/type-i-and-type-ii-errors.jpg

with plt.xkcd():
    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.linspace(-5, 5, 200)
    y0 = stats.norm.pdf(x, 0, 1)
    y1 = stats.norm.pdf(x, 1, 1)
    mask = x >= 2
    ax.plot(x, y0, x, y1, '-')
    ax.fill_between(x[mask], y0[mask], alpha=0.5)
    ax.fill_between(x[~mask], y1[~mask], alpha=0.5)
    ax.vlines(2, 0, 0.5, linestyle='--', lw=2)
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    ax.set_xlabel("$q$")
    ax.annotate("$f(q|H_1)$", xy=(x[100], y1[100]), xytext=(0.2, 0.7), textcoords="figure fraction", size=20,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), color='green')
    ax.annotate("$f(q|H_0)$", xy=(x[90], y0[90]), xytext=(0.3, 0.8), textcoords="figure fraction", size=20,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), color='blue')
    ax.annotate(r"$\beta$", xy=(x[100], 0.1), xytext=(0.1, 0.5), textcoords="figure fraction", size=20,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), color='green')
    ax.annotate(r"$\alpha$", xy=(x[142], 0.01), xytext=(0.8, 0.25), textcoords="figure fraction", size=20,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), color='blue')
    ax.text(2, 0.4, r'$q_\alpha$', rotation='vertical', ha='right', size=20)
plt.close()

fig

N = np.arange(1, 70)
alpha = np.sort(np.concatenate((np.logspace(-8, -3, 20), np.arange(0.001, 0.2, 0.002))))

kalpha = stats.binom.isf(alpha.reshape(len(alpha), 1), N, 0.5)
results_must_guess = kalpha.T + 1
power = stats.binom.sf(kalpha, N, 0.9).T
min_N = np.ceil(np.log(alpha) / np.log(0.5))

from mpl_toolkits.axes_grid1 import make_axes_locatable


fig, ax = plt.subplots(1, 2, figsize=(15, 6))
p = ax[1].pcolormesh(alpha, N, power, vmin=0, vmax=1, cmap='viridis')
div = make_axes_locatable(ax[1])
cax = div.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(p, cax=cax)
ax[0].set_xscale('log')
ax[1].set_xscale('log')
p = ax[0].pcolormesh(alpha, N, results_must_guess, vmin=0, cmap='viridis')
div = make_axes_locatable(ax[0])
cax = div.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(p, cax=cax)

ax[0].fill_between(alpha, min_N, color='white')
ax[1].fill_between(alpha, min_N, color='white')

ax[1].set_title('power')
ax[0].set_title('minimum corrected guess to reject H0', fontsize=15)
ax[1].set_xlabel('alpha')
ax[1].set_ylabel('number of cups')
ax[0].set_xlabel('alpha')
ax[0].set_ylabel('number of cups')
plt.show()

fig

ws = ROOT.RooWorkspace()
true_density = ws.factory("true_density[2, 3]")
ws.factory("Gaussian::pdf1(density1[0,50], true_density, resolution1[0.5])")
ws.factory("Gaussian::pdf2(density2[0,50], true_density, resolution2[0.2])")
pdf = ws.factory("PROD::pdf(pdf1, pdf2)")

pdf_opal = ws.factory("EDIT:pdf_opal(pdf, true_density=density_opal[2.2])")
pdf_quartz = ws.factory("EDIT:pdf_quartz(pdf, true_density=density_quartz[2.6])")

# do the experiment, use H1 to generate (opal)
data = pdf_opal.generate(ROOT.RooArgSet(ws.var('density1'),
                                        ws.var('density2')), 1)  # generate with 1 entry
data.get(0).Print("V")

opal_model = RooStats.ModelConfig("opal (H1)", ws)   
opal_model.SetPdf(pdf_opal)
opal_model.SetParametersOfInterest('density_opal')  # not useful
opal_model.SetObservables('density1,density2')  # no space
opal_model.SetSnapshot(ROOT.RooArgSet(ws.var('density_opal')))

quartz_model = opal_model.Clone("quartz (H0)")
quartz_model.SetParametersOfInterest('density_quartz')  # not useful
quartz_model.SetPdf(pdf_quartz)

# define the test-statistic

test = RooStats.SimpleLikelihoodRatioTestStat(pdf_quartz, pdf_opal)  # null, alt
test.EnableDetailedOutput(True)

hypoCalc = RooStats.FrequentistCalculator(data, opal_model, quartz_model)  # alt, null
hypoCalc.SetToys(20000, 1000)  # generate much more toys for bkg-only

toy_sampler = hypoCalc.GetTestStatSampler()
toy_sampler.SetTestStatistic(test)  # our test statistics
toy_sampler.SetNEventsPerToy(1)

htr = hypoCalc.GetHypoTest()
htr.SetPValueIsRightTail(True)  # the "extreme" cases are on the right tail
htr.SetBackgroundAsAlt(False)

plot = RooStats.HypoTestPlot(htr, 100, -15, 15)
canvas = ROOT.TCanvas()
plot.Draw()
canvas.Draw()

print "observed qvalue: ", test.Evaluate(data, ROOT.RooArgSet(ws.var('density_quartz')))
print "observed p-value: ", htr.NullPValue()
alpha = 0.05
kalpha = htr.GetNullDistribution().InverseCDF(1 - alpha)
beta = htr.GetAltDistribution().CDF(kalpha)
print "k_alpha = ", kalpha
print "power = ", 1 - beta

def eval_test_stat():
    d = ROOT.RooDataSet("d", "d", ROOT.RooArgSet(ws.var('density1'), ws.var('density2')))
    d.add(ROOT.RooArgSet(ws.var('density1'), ws.var('density2')))
    return test.Evaluate(d, ROOT.RooArgSet(ws.var('true_density')))

fig_opal, ax = plt.subplots(figsize=(12, 7))

x = np.linspace(1.3, 3.5, 100)
y = np.linspace(1.5, 3.5, 100)

qvalues = np.zeros((len(x), len(y)))
pdf_opal_values = np.zeros((len(x), len(y)))
pdf_quartz_values = np.zeros((len(x), len(y)))
for ix, xx in enumerate(x):
    for iy, yy in enumerate(y):        
        ws.var('density1').setVal(xx)
        ws.var('density2').setVal(yy)
        qvalues[iy, ix] = eval_test_stat()
        pdf_opal_values[iy, ix] = pdf_opal.getVal()
        pdf_quartz_values[iy, ix] = pdf_quartz.getVal()
        
c = plt.pcolormesh(x, y, qvalues, cmap=colormaps.viridis)
cs = ax.contour(x, y, qvalues, [kalpha], linestyles='dashed')
ax.clabel(cs, fontsize=14, fmt=r"$k_\alpha$ = %1.3f", inline_spacing=30, use_clabeltext=True)
ax.contour(x, y, pdf_opal_values, cmap='Blues', alpha=0.6)
ax.contour(x, y, pdf_quartz_values, cmap='Oranges', alpha=0.6)
ax.plot(ws.var('density_opal').getVal(), ws.var('density_opal').getVal(), 'o', label='opal (H1)')
ax.plot(ws.var('density_quartz').getVal(), ws.var('density_quartz').getVal(), 'o', label='quartz (H0)')
ax.text(1.5, 1.6, "reject", size=20)
ax.text(1.8, 3.2, "accept", size=20)
datalist = [x.getVal() for x in iter_collection(data.get(0))]
ax.plot(datalist[0], datalist[1], 'v', label='observed', color='red')

ax.set_xlabel('density1')
ax.set_ylabel('density2')
ax.legend()
ax.set_aspect('equal')
plt.colorbar(c)
plt.close()

fig_opal

alphas = np.linspace(0, 1, 100)
powers = []

for alpha in alphas:
    kalpha = htr.GetNullDistribution().InverseCDF(1 - alpha)
    beta = htr.GetAltDistribution().CDF(kalpha)
    powers.append(1 - beta)
    
plt.plot(alphas, powers)   
plt.xlabel(r'$\alpha$')
plt.ylabel(r'power')
plt.show()

NOBS = 4
ALPHA = z2p(1)  # 0.158...

def compute_pvalue(nobs, ntrue, left):
    if left:
        return stats.poisson(ntrue).cdf(nobs)
    else:
        return stats.poisson(ntrue).sf(nobs) + stats.poisson(ntrue).pmf(nobs)

fig, axs = plt.subplots(1, 3, figsize=(13, 4))
x = np.linspace(NOBS, NOBS + 3 * NOBS, 20)

axs[1].plot(x, compute_pvalue(NOBS, x, left=True), 'r')
axs[1].hlines(ALPHA, np.min(x), np.max(x), linestyles='--')

x = np.linspace(max(NOBS - NOBS * 3, 0), NOBS, 20)
axs[0].plot(x, compute_pvalue(NOBS, x, left=False), 'b')
axs[0].hlines(ALPHA, np.min(x), np.max(x), linestyles='--')

from scipy.optimize import brenth
nup = brenth(lambda x: compute_pvalue(NOBS, x, True) - ALPHA, NOBS, NOBS * 3)
ndown = brenth(lambda x: compute_pvalue(NOBS, x, False) - ALPHA, 1E-5, NOBS)

axs[1].plot(nup, compute_pvalue(NOBS, nup, True), 'ro')
axs[0].plot(ndown, compute_pvalue(NOBS, ndown, False), 'bo')
axs[1].set_xlabel('N'); axs[0].set_xlabel('N')
axs[1].set_ylabel('$P(n\leq n_{obs})$')
axs[0].set_ylabel('$P(n\geq n_{obs})$')
plt.close()

x = np.arange(0, NOBS * 3)
axs[2].plot(x, stats.poisson(nup).pmf(x), 'r', linestyle='steps',)
axs[2].plot(x, stats.poisson(ndown).pmf(x), 'b', linestyle='steps')
axs[2].vlines(NOBS, 0, 0.3, linestyles='--')

fig

print ndown, nup
Latex("$\lambda = %.2f^{+%.2f}_{-%.2f}$" % (NOBS, nup - NOBS, NOBS - ndown))

def get_confiderence_poisson(n):
    ym = np.array([0], dtype=np.float64)
    yp = np.array([0], dtype=np.float64)
    ROOT.RooHistError.instance().getPoissonInterval(n, ym, yp, 1)
    return ym[0], yp[0]

get_confiderence_poisson(4)

fig, ax = plt.subplots(figsize=(7, 5))
trues = np.linspace(1E-5, NOBS * 4, 100)
obss = np.arange(0, NOBS * 3)
v = np.zeros((len(trues), len(obss)))
limits = []
for ix, obs in enumerate(obss):
    limits.append(get_confiderence_poisson(obs))   
    for iy, true in enumerate(trues):
        v[iy, ix] = stats.poisson(true).pmf(obs)
ax.pcolormesh(obss, trues, v, vmax=np.percentile(v, 99))
ax.plot(obss, np.array(limits).T[0], 'k', drawstyle='steps-post')
ax.plot(obss, np.array(limits).T[1], 'k', drawstyle='steps-post')
ax.set_xlim(0, np.max(obss))
ax.set_xlabel('obs')
ax.set_ylabel('$\lambda$'); plt.show()

def f(true):
    return stats.poisson(true).pmf(NOBS)

import scipy
from scipy.optimize import brenth, brentq
def f_low(x):
    return scipy.integrate.quad(f, 0, x)[0] - z2p(1)

def f_hi(x):
    return scipy.integrate.quad(f, x, NOBS * 10)[0] - z2p(1)

lo_bayes, hi_bayes = brentq(f_low, 1E-5, NOBS), brentq(f_hi, NOBS, NOBS * 5)
print lo_bayes, hi_bayes
Latex("$\lambda = %.2f_{-%.2f}^{+%.2f}$" % (NOBS, NOBS - lo_bayes, hi_bayes - NOBS))

def invert(f, y, a, b):
    return brentq(lambda x: f(x) - y, a, b)

lo_bayes_el = brenth(
    lambda x: scipy.integrate.quad(f, x, invert(f, f(x), x * (1 + 1E-5), NOBS * 20))[0] - (1 - z2p(1) * 2),
    1E-2, NOBS - 1E-2)
hi_bayes_el = invert(f, f(lo_bayes_el), NOBS, NOBS * 5)
print lo_bayes_el, hi_bayes_el

def width_interval(x):
    return brenth(lambda y: scipy.integrate.quad(f, x, y)[0] - (1 - z2p(1) * 2), x, NOBS * 20) - x

sol = scipy.optimize.minimize_scalar(width_interval, (1E-1, 3), bounds=(0, NOBS), method='bounded')
lo_bayes_min, hi_bayes_min = sol.x, sol.x + width_interval(sol.x)
print lo_bayes_min, hi_bayes_min

x = np.linspace(0, NOBS * 3, 100)

fig, axs = plt.subplots(1, 3, figsize=(14, 4))

axs[0].plot(x, f(x))
axs[0].fill_between(np.linspace(0, lo_bayes, 50), f(np.linspace(0, lo_bayes, 50)))
axs[0].fill_between(np.linspace(hi_bayes, NOBS * 3, 50), f(np.linspace(hi_bayes, NOBS * 3, 50)))
axs[0].set_title('equal probability (central)', fontsize=15)
axs[0].text(7, 0.16, '$\lambda = %.2f_{-%.2f}^{+%.2f}$' % (NOBS, NOBS - lo_bayes, hi_bayes - NOBS), size=15)
axs[0].text(4, 0.04, '68%', size=15)
axs[0].set_xlabel('$\lambda$')
axs[1].plot(x, f(x))
axs[1].fill_between(np.linspace(0, lo_bayes_el, 50), f(np.linspace(0, lo_bayes_el, 50)))
axs[1].fill_between(np.linspace(hi_bayes_el, NOBS * 3, 50), f(np.linspace(hi_bayes_el, NOBS * 3, 50)))
axs[1].set_title('equal likelihood', fontsize=15)
axs[1].text(7, 0.16, '$\lambda = %.2f_{-%.2f}^{+%.2f}$' % (NOBS, NOBS - lo_bayes_el, hi_bayes_el - NOBS), size=15)
axs[1].text(3.5, 0.04, '68%', size=15)
axs[1].set_xlabel('$\lambda$')
axs[2].plot(x, f(x))
axs[2].fill_between(np.linspace(0, lo_bayes_min, 50), f(np.linspace(0, lo_bayes_min, 50)))
axs[2].fill_between(np.linspace(hi_bayes_min, NOBS * 3, 50), f(np.linspace(hi_bayes_min, NOBS * 3, 50)))
axs[2].set_title('minimum interval', fontsize=15)
axs[2].text(7, 0.16, '$\lambda = %.2f_{-%.2f}^{+%.2f}$' % (NOBS, NOBS - lo_bayes_min, hi_bayes_min - NOBS), size=15)
axs[2].text(3.5, 0.04, '68%', size=15)
axs[2].set_xlabel('$\lambda$')
for ax in axs: ax.set_yticklabels([])
plt.close()

fig

S, B = 30, 200  # expected values

ws_poisson = ROOT.RooWorkspace('ws_poisson')
b = ws_poisson.factory("b[%f]" % B)
n_exp = ws_poisson.factory("sum::n_exp(s[%f, -500, 500], b)" % S)
pdf = ws_poisson.factory("Poisson::pdf(n_obs[0, 1000], n_exp)")
aset = ROOT.RooArgSet(ws_poisson.var('n_obs'))
data = pdf.generate(aset, 1)  # generate with 1 entry
print "observed = ", data.get(0).first().getVal()

# create a MC linked to the ws
sbModel = RooStats.ModelConfig("sbmodel", ws_poisson)   
sbModel.SetPdf(pdf)
sbModel.SetObservables('n_obs')  # it understands variable-names
sbModel.SetParametersOfInterest('s')
# save the value of s for (s+b)-hypothesis
sbModel.SetSnapshot(ROOT.RooArgSet(ws_poisson.var('s')))
getattr(ws_poisson, 'import')(sbModel)  # import is a keyword in python

bModel = sbModel.Clone("bmodel")  # create a bkg-only model
ws_poisson.var('s').setVal(0)             # with no signal
bModel.SetSnapshot(ROOT.RooArgSet(ws_poisson.var('s')))
getattr(ws_poisson, 'import')(bModel);

profll = RooStats.ProfileLikelihoodTestStat(bModel.GetPdf())
# this modify a bit our test statistics
profll.SetOneSidedDiscovery(1)

hypoCalc = RooStats.FrequentistCalculator(data, sbModel, bModel)
hypoCalc.SetToys(100000, 5000)  # generate much more toys for bkg-only

toy_sampler = hypoCalc.GetTestStatSampler()
toy_sampler.SetTestStatistic(profll)
toy_sampler.SetNEventsPerToy(1)

htr = hypoCalc.GetHypoTest()
htr.SetPValueIsRightTail(True)  # the "extreme" cases are on the right tail
htr.SetBackgroundAsAlt(False)

plot = RooStats.HypoTestPlot(htr, 50, 0, 10)
canvas = ROOT.TCanvas()
plot.Draw()
canvas.SetLogy(); canvas.Draw()
print "pvalue = ", htr.NullPValue(), " significance = ", htr.Significance()

hypoCalc = RooStats.AsymptoticCalculator(data, sbModel, bModel)
hypoCalc.SetOneSidedDiscovery(True)
htr_asym = hypoCalc.GetHypoTest()
print "pvalue =", htr_asym.NullPValue(), " significance =", htr_asym.Significance()

ws_onoff = ROOT.RooWorkspace('ws_onoff')
model_sr = ws_onoff.factory("Poisson:N_SR(n_sr[0, 5000], sum:s_plus_b(s[15, 0, 100], b[50, 0, 100]))")
model_cr = ws_onoff.factory("Poisson:N_CR(n_cr[0, 5000], prod:alpha_x_b(alpha[10, 0, 10], b))")
model = ws_onoff.factory("PROD:model(N_SR, N_CR)")
ws_onoff.var("alpha").setConstant(True)

sbModel = RooStats.ModelConfig('sbModel', ws_onoff)
sbModel.SetObservables('n_sr,n_cr')
sbModel.SetParametersOfInterest('s')
sbModel.SetPdf('model')
sbModel.SetSnapshot(ROOT.RooArgSet(ws_onoff.var('s')))
getattr(ws_onoff, 'import')(sbModel)

bModel = sbModel.Clone("bModel")
ws_onoff.var('s').setVal(0)
bModel.SetSnapshot(bModel.GetParametersOfInterest())

ws_onoff.Print()
ws_onoff.writeToFile('onoff.root')

model.graphVizTree("on_off_graph.dot")
get_ipython().system('dot -Tsvg on_off_graph.dot > on_off_graph.svg; rm on_off_graph.dot')
SVG("on_off_graph.svg")

sbModel.LoadSnapshot()
data = model.generate(bModel.GetObservables(), 1)
print "observed  N_SR = %.f, N_CR = %.f" % tuple([x.getVal() for x in iter_collection(data.get(0))])
model.fitTo(data)
print "best fit     s        b"
print "SR    {:>8.1f} {:>8.1f}".format(ws_onoff.var('s').getVal(), ws_onoff.var('b').getVal())
print "CR             {:>8.1f}".format(ws_onoff.function('alpha_x_b').getVal())

profll = RooStats.ProfileLikelihoodTestStat(bModel.GetPdf())
# this modify a bit our test statistics
profll.SetOneSidedDiscovery(True)

hypoCalc = RooStats.FrequentistCalculator(data, sbModel, bModel)
hypoCalc.SetToys(10000, 500)

toy_sampler = hypoCalc.GetTestStatSampler()
toy_sampler.SetTestStatistic(profll)
toy_sampler.SetNEventsPerToy(1)

htr = hypoCalc.GetHypoTest()
htr.SetPValueIsRightTail(True)  # the "extreme" cases are on the right tail
htr.SetBackgroundAsAlt(False)

plot = RooStats.HypoTestPlot(htr, 50, 0, 20)
canvas = ROOT.TCanvas()
plot.Draw()
canvas.SetLogy()
canvas.Draw()
print "pvalue = ", htr.NullPValue(), " significance = ", htr.Significance()

# create profiled log-likelihood
prof = model.createNLL(data).createProfile(ROOT.RooArgSet(ws_onoff.var('s')))
# multiply by 2
minus2LL = ROOT.RooFormulaVar("minus2LL", "2 * @0", ROOT.RooArgList(prof))
frame = ws_onoff.var('s').frame(0, 30)
minus2LL.plotOn(frame)
frame.SetYTitle("-2 log#Lambda(s)")
canvas = ROOT.TCanvas()
frame.Draw()
canvas.Draw()

hypoCalc = RooStats.AsymptoticCalculator(data, sbModel, bModel)
hypoCalc.SetOneSidedDiscovery(True)
htr = hypoCalc.GetHypoTest()
print "pvalue =", htr.NullPValue(), " significance =", htr.Significance()

pvalue_exp = RooStats.AsymptoticCalculator.GetExpectedPValues(htr.NullPValue(), htr.AlternatePValue(), False, True)
significance_exp = ROOT.Math.normal_quantile_c(pvalue_exp, 1)
print "expected p-value = ", pvalue_exp, " significance =", significance_exp                



