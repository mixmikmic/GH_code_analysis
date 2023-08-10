import ROOT

f = ROOT.TFile.Open("/home/adminuser/Dropbox/Public/CEVALE_Tarea_3/input_MC_samples_Tarea4_cevale2ven.root")

t = f.Get("tree")

h = ROOT.TH1F("variable","Example variable from the mini tree",40,0,200)

c = ROOT.TCanvas("testCanvas","a first way to plot a variable",800,600)
t.Draw("lepZ_m:realZ_LJ_m","lepZ_m>0.","colz")

c.Draw()

for event in t:
    if t.RunNumber==167809 or t.RunNumber==167812:
        h.Fill(t.lepZ_m/1000)

print "Done!"
    

h.Draw()
c.Draw()

scale = h.Integral()
h.Scale(1/scale)
h.Draw()
c.Draw()

h.Fit("gaus","S")

## h.Fit("gaus","S")

get_ipython().run_cell_magic('cpp', '', 'variable->SetFillColor(kBlue);\nvariable->Fit("gaus","S");\ntestCanvas->Draw();')





