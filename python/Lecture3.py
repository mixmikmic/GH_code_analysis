from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
from IPython.display import HTML, IFrame, Image, SVG, Latex
import re
import ROOT
from ROOT import RooFit, RooStats
get_ipython().magic('matplotlib inline')
#%matplotlib nbagg
#%matplotlib notebook
from ipywidgets import interact, interactive, fixed
import colormaps

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

#ws = ws_onoff_withsys = ws_onoff.Clone("ws_onoff_withsys")
f = ROOT.TFile.Open("onoff.root")
ws_onoff = f.Get("ws_onoff")

# create the term kalpha = (1 + sigma * theta) with a relative error of 20%
ws_onoff.factory('expr:kalpha("1 + @0 * @1", {sigma_alpha[0.2], theta_alpha[0, -5, 5]})')
ws_onoff.factory('prod:alpha_x_kappa(alpha, kalpha)')
# create new pdf model replacing alpha -> alpha_x_kalpha
ws_onoff.factory('EDIT:model_with_sys(model, alpha=alpha_x_kappa)')

# create new workspace
ws_onoff_sys = ROOT.RooWorkspace('ws_onoff_sys')
getattr(ws_onoff_sys, 'import')(ws_onoff.pdf('model_with_sys'))
# create the constraint
ws_onoff_sys.factory("Gaussian:constraint_alpha(global_alpha[0, -5, 5], theta_alpha, 1)")
ws_onoff_sys.var("global_alpha").setConstant(True)
# final pdf
model = ws_onoff_sys.factory("PROD:model_constrained(model_with_sys, constraint_alpha)")

ws_onoff_sys.Print()

model.graphVizTree("on_off_with_sys_graph.dot")
get_ipython().system('dot -Tsvg on_off_with_sys_graph.dot > on_off_with_sys_graph.svg; rm on_off_with_sys_graph.dot')
s = SVG("on_off_with_sys_graph.svg")
s.data = re.sub(r'width="[0-9]+pt"', r'width="90%"', s.data)
s.data = re.sub(r'height="[0-9]+pt"', r'height=""', s.data); s

sbModel = ROOT.RooStats.ModelConfig('sbModel_sys', ws_onoff_sys)
sbModel.SetPdf('model_constrained')
sbModel.SetParametersOfInterest('s')
sbModel.SetObservables('n_sr,n_cr')
sbModel.SetNuisanceParameters('theta_alpha')
ws_onoff_sys.var('s').setVal(30)
sbModel.SetSnapshot(ROOT.RooArgSet(ws_onoff_sys.var('s')))
getattr(ws_onoff_sys, 'import')(sbModel)

bModel = sbModel.Clone("bModel_sys")
ws_onoff_sys.var('s').setVal(0)
bModel.SetSnapshot(bModel.GetParametersOfInterest())
getattr(ws_onoff_sys, 'import')(bModel)

sbModel.LoadSnapshot()
ws_onoff_sys.var('s').Print()
data = model.generate(bModel.GetObservables(), 1)
data.SetName('obsData')
print "observed  N_SR = %.f, N_CR = %.f" % tuple([x.getVal() for x in iter_collection(data.get(0))])
model.fitTo(data)
print "best fit"
print "SR {:>8.1f} {:>8.1f}".format(ws_onoff_sys.var('s').getVal(), ws_onoff_sys.var('b').getVal())
print "CR          {:>8.1f}".format(ws_onoff_sys.function('alpha_x_b_model_with_sys').getVal())
getattr(ws_onoff_sys, 'import')(data)
ws_onoff_sys.writeToFile('onoff_sys.root')

# create profiled log-likelihood as a function of s
ws_onoff_sys.var('theta_alpha').setConstant(False)
prof = model.createNLL(data).createProfile(ROOT.RooArgSet(ws_onoff_sys.var('s')))
# multiply by 2
minus2LL = ROOT.RooFormulaVar("minus2LL", "2 * @0", ROOT.RooArgList(prof))
frame = ws_onoff.var('s').frame(0, 60)
minus2LL.plotOn(frame)

ws_onoff_sys.var('theta_alpha').setConstant(True)
minus2LL.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kRed))
frame.SetYTitle("-2 log#Lambda(s)")

canvas = ROOT.TCanvas()
frame.Draw()
canvas.Draw()

ws_onoff_sys.var('theta_alpha').setConstant(False)
hypoCalc = RooStats.AsymptoticCalculator(data, sbModel, bModel)
hypoCalc.SetOneSidedDiscovery(True)
htr = hypoCalc.GetHypoTest()
print "pvalue =", htr.NullPValue(), " significance =", htr.Significance()

ws = ROOT.RooWorkspace("ws_shape")
mH = ws.factory("mH[125, 90, 150]")  # true mass of the resonance
mass = ws.factory("mass[80, 160]")    # observed invariant mass
ws.factory('expr:kpeak("1 + @0 * @1", {sigma_mH[0.01], theta_mH[0, -5, 5]})')
peak = ws.factory('expr:peak("@0 * @1", {mH, kpeak})')   # peak position for signal
ws.factory('expr:kwidth("1 + @0 * @1", {sigma_width[0.05], theta_width[0, -5, 5]})')
width = ws.factory('expr:width("@0 * @1", {nominal_width[5], kwidth})')
signal = ws.factory("RooGaussian:signal(mass, peak, width)")
ws.factory("RooExponential:background(mass, tau[-0.03, -0.5, -0.001])")
ws.factory("nbkg[400, 0, 1000]")
ws.factory('expr:klumi("(1 + exp(@0 * @1))", {sigma_lumi[0.02], theta_lumi[0, -5, 5]})')
ws.factory('expr:efficiency("@0 * (1 + exp(@1 * @2))", {nominal_efficiency[0.6], sigma_efficiency[0.05], theta_efficiency[0, -5, 5]})')
ws.factory('expr:nsignal_theory("7 + @0 * 0.2", {mH})')
ws.factory('prod:nsignal(nsignal_theory, mu[1, -2, 5], efficiency, klumi)')
ws.factory("SUM:phys_pdf(nsignal * signal, nbkg * background)")
ws.factory("RooGaussian:constrain_peak(global_peak[0, -5, 5], theta_mH, 1)")
ws.factory("RooGaussian:constrain_width(global_width[0, -5, 5], theta_width, 1)")
ws.factory("RooGaussian:constrain_lumi(global_lumi[0, -5, 5], theta_lumi, 1)")
ws.factory("RooGaussian:constrain_eff(global_efficiency[0, -5, 5], theta_efficiency, 1)")
ws.factory("PROD:constraints(constrain_peak, constrain_lumi, constrain_width, constrain_eff)")
model = ws.factory("PROD:model(phys_pdf, constraints)")

model.graphVizTree("shape_graph.dot")
get_ipython().system('dot -Tsvg shape_graph.dot > shape_graph.svg; rm shape_graph.dot')
s = SVG("shape_graph.svg")
s.data = re.sub(r'width="[0-9]+pt"', r'width="90%"', s.data)
s.data = re.sub(r'height="[0-9]+pt"', r'height=""', s.data)
s

RooStats.SetAllConstant(ws.allVars().selectByName('global*'))

data = model.generate(ROOT.RooArgSet(mass))
data.SetName('obsData')
getattr(ws, 'import')(data)

ROOT.RooMsgService.instance().setGlobalKillBelow(5)
fit_result = model.fitTo(data, ROOT.RooFit.Save(), RooFit.PrintLevel(0))

ROOT.RooMsgService.instance().setGlobalKillBelow(2)
fit_result.Print()

frame = mass.frame(50)
data.plotOn(frame)
model.plotOn(frame)
frame.Draw()
ROOT.gPad.Draw()

sbModel = RooStats.ModelConfig('sbModel', ws)
sbModel.SetPdf('model')
nps = ws.allVars().selectByName('theta*')
nps.add(ws.var('tau'))
nps.add(ws.var('nbkg'))
sbModel.SetNuisanceParameters(nps)
sbModel.SetObservables('mass')
sbModel.SetParametersOfInterest('mu')
sbModel.SetGlobalObservables(ws.allVars().selectByName('global*'))
sbModel.SetSnapshot(ROOT.RooArgSet(ws.var('mu')))

bModel = sbModel.Clone('bModel')
ws.var('mu').setVal(0)
bModel.SetSnapshot(ROOT.RooArgSet(ws.var('mu')))

getattr(ws, 'import')(sbModel)
getattr(ws, 'import')(bModel)

ws.writeToFile('ws_shape.root')

RooStats.AsymptoticCalculator.SetPrintLevel(-1)
mH_values = np.linspace(mH.getMin(), mH.getMax(), 20)
pvalues, pvalues_exp, zs, qvalues = [], [], [], []
for mH_value in mH_values:
    f = ROOT.TFile('ws_shape.root')
    ws = f.Get('ws_shape')
    mH = ws.var('mH')
    mH.setVal(mH_value)
    mH.setConstant(True)
    data = ws.data('obsData')
    #ws.pdf('model').fitTo(data)   # better to do a fit before
    sbModel = ws.obj('sbModel')
    bModel = ws.obj('bModel')
    hypoCalc = RooStats.AsymptoticCalculator(data, sbModel, bModel)
    hypoCalc.SetOneSidedDiscovery(True)
    htr = hypoCalc.GetHypoTest()
    print "mH = ", mH.getVal(), "pvalue =", htr.NullPValue(), " significance =", htr.Significance()
    pval_exp = RooStats.AsymptoticCalculator.GetExpectedPValues(htr.NullPValue(), htr.AlternatePValue(), 0, False)
    pvalues.append(htr.NullPValue()); zs.append(htr.Significance()); pvalues_exp.append(pval_exp)
    del hypoCalc
    del htr

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].semilogy(mH_values, pvalues, '.-')
axs[0].semilogy(mH_values, pvalues_exp, '--')
axs[1].plot(mH_values, zs, '.-')
axs[0].set_xlabel('mH'); axs[1].set_xlabel('mH')
axs[0].set_ylabel('pvalue'); axs[1].set_ylabel('significance'); plt.show()

IFrame("http://xkcd.com/882/", 900, 500)

f = ROOT.TFile('ws_shape.root')
ws = f.Get("ws_shape")
ws.var('mH').setVal(110)
ws.var('mH').setConstant(True)
ws.writeToFile("ws_shape_110.root")

ROOT.gROOT.LoadMacro('StandardHypoTestInvDemo.C')
ROOT.StandardHypoTestInvDemo("ws_shape_110.root", "ws_shape", "sbModel", "bModel", "obsData", 2, 3, True, 30)

fresult = ROOT.TFile("Asym_CLs_grid_ts3_ws_shape_110.root")
result = fresult.Get("result_mu")
plot = RooStats.HypoTestInverterPlot("result", "result", result)
plot.Draw()
ROOT.gPad.Draw()

IFrame("figaux_004a.pdf", width="90%", height="700px")

IFrame("figaux_007a.pdf", width="90%", height="600px")

IFrame("figaux_007b.pdf", width="90%", height="600px")

