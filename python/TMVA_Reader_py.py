import ROOT
from ROOT import TMVA

from IPython.core.extensions import ExtensionManager
ExtensionManager(get_ipython()).load_extension("JsMVA.JsMVAMagic")

get_ipython().magic('jsroot on')

methodName = "BDT"

inputFile = ROOT.TFile("Higgs_data.root")

inputFile.ls()
#inputFile.sig_tree.Print()

TMVA.Tools.Instance()

reader = TMVA.Reader( "!Color:!Silent" )

from array import array
m_jj = array('f',[0])
m_jjj = array('f',[0])
m_lv = array('f',[0])
m_jlv = array('f',[0])
m_bb = array('f',[0])
m_wbb = array('f',[0])
m_wwbb = array('f',[0])

#add variables 
reader.AddVariable("m_jj",m_jj)
reader.AddVariable("m_jjj",m_jjj)
reader.AddVariable("m_lv",m_lv)
reader.AddVariable("m_jlv",m_jlv)
reader.AddVariable("m_bb",m_bb)
reader.AddVariable("m_wbb",m_wbb)
reader.AddVariable("m_wwbb",m_wwbb)

weightfile = "dataset/weights/TMVAClassification_" + methodName + ".weights.xml"

reader.BookMVA( methodName, weightfile );

h1 = ROOT.TH1D("h1","Classifier Output on Background Events",100,-0.5,0.5)
h2 = ROOT.TH1D("h2","Classifier Output on Signal Events",100,-0.5,0.5)

ievt = 0

for entry in inputFile.bkg_tree:
    
    m_jj[0] = entry.m_jj
    m_jjj[0] = entry.m_jjj
    m_lv[0] = entry.m_lv
    m_jlv[0] = entry.m_jlv
    m_bb[0] = entry.m_bb
    m_wbb[0] = entry.m_wbb
    m_wwbb[0] = entry.m_wwbb
    
    output = reader.EvaluateMVA(methodName)
  
    h1.Fill(output)
    
    
    if (ievt%10000)==0 : print 'Event ',ievt,'m_jj=',m_jj[0],'MVA output =',output
    ievt += 1
#    if (ievt > 20000) : break
    

h1.Draw()
ROOT.gPad.Draw()

ievt = 0

for entry in inputFile.sig_tree:

    m_jj[0] = entry.m_jj
    m_jjj[0] = entry.m_jjj
    m_lv[0] = entry.m_lv
    m_jlv[0] = entry.m_jlv
    m_bb[0] = entry.m_bb
    m_wbb[0] = entry.m_wbb
    m_wwbb[0] = entry.m_wwbb
    
    output = reader.EvaluateMVA(methodName)
  
    h2.Fill(output)
    
    if (ievt%10000)==0 : print 'Event ',ievt,'m_jj=',m_jj[0],'MVA output =',output
    ievt += 1
#    if (ievt > 10000) : break
    

h2.SetLineColor(ROOT.kRed)

h2.Draw("SAME")
ROOT.gPad.Draw()
ROOT.gPad.BuildLegend()



