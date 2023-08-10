import ROOT

h = ROOT.TH1F("gauss","Example histogram",64,-4,4)
h.FillRandom("gaus")

c = ROOT.TCanvas("myCanvasName","The Canvas Title",800,600)
h.Draw()

c.Draw()

get_ipython().magic('jsroot on')
c.Draw()

outputFile = ROOT.TFile("output.root","RECREATE")
h.Write()
outputFile.Close()

get_ipython().run_cell_magic('bash', '', 'rootls -l output.root')

get_ipython().run_cell_magic('cpp', '', 'cout << "This is a C++ cell" << endl;')

get_ipython().run_cell_magic('cpp', '', 'class A{\n    public:\n    A(){cout << "Constructor of A!" << endl;}\n};')

a = ROOT.A()

get_ipython().run_cell_magic('cpp', '', 'gauss->Fit("gaus", "S");\nmyCanvasName->Draw();')

get_ipython().magic('pinfo %%cpp')

get_ipython().run_cell_magic('cpp', '-d', 'void f() {\n    cout << "This is function f" << endl;\n}')

print "This is again Python"
ROOT.f()

get_ipython().run_cell_magic('cpp', '', 'f')

get_ipython().run_cell_magic('cpp', '-a', 'class CompileMe {\npublic:\n    CompileMe() {}\n    void run() {}\n};')

ROOT.TClass.GetClass("CompileMe").HasDictionary()

ROOT.TClass.GetClass("A").HasDictionary()

