import ROOT

get_ipython().run_cell_magic('cpp', '-d', 'auto pi = TMath::Pi();\ndouble single(double *x, double *par){return pow(sin(pi*par[0]*x[0])/(pi*par[0]*x[0]),2);};\ndouble nslit0(double *x, double *par){return pow(sin(pi*par[1]*x[0])/sin(pi*x[0]),2);};\ndouble nslit(double *x, double *par){return single(x,par) * nslit0(x,par);};')

interfTF1 = ROOT.TF1("Slits interference",ROOT.nslit,-5.001,5.,2)
interfTF1.SetNpx(1000)
c = ROOT.TCanvas("c","c",1600,1200)

from ipywidgets import interact, FloatSlider

RatioSlider = FloatSlider(min=.05, max=1., step=0.05, value=0.2)
NSSlider = FloatSlider(min=1, max=10, step=1, value=2)
@interact(Ratio = RatioSlider, Number_Of_Slits = NSSlider)
def interFunction(Ratio, Number_Of_Slits):
    interfTF1.SetParameters(Ratio,Number_Of_Slits)
    interfTF1.Draw()
    c.Draw()
    return 0

