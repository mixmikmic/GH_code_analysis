import ROOT
from PyTreeReader import PyTreeReader

h = ROOT.TH1F("h","h",1024,-256,256)
fill = h.Fill

f = ROOT.TFile(ROOT.gROOT.GetTutorialsDir()+"/hsimple.root")
tree = f.ntuple

get_ipython().run_cell_magic('timeit', '-n 1 -r 1', 'for event in tree:\n    fill(event.px*event.py*event.pz*event.random)')

get_ipython().run_cell_magic('timeit', '-n 1 -r 1', 'for event in PyTreeReader(tree):\n    fill(event.px()*event.py()*event.pz()*event.random())')

get_ipython().run_cell_magic('timeit', '-n 1 -r 1', 'ptr = PyTreeReader(tree, cache=True)')

ptr = PyTreeReader(tree, cache=True)

get_ipython().run_cell_magic('timeit', '-n 1 -r 1', 'for event in ptr:\n    fill(event.px()*event.py()*event.pz()*event.random())')

get_ipython().run_cell_magic('timeit', '-n 1 -r 1', 'ptr = PyTreeReader(tree, cache=True, pattern="p[*]")')

for py in ptr.py_array()[:10]: print py # Better stopping after a few of them :)

ptr.py_array()

