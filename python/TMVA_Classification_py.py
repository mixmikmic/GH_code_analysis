import ROOT
from ROOT import TMVA

from IPython.core.extensions import ExtensionManager
ExtensionManager(get_ipython()).load_extension("JsMVA.JsMVAMagic")

get_ipython().magic('jsmva on')

inputFile = ROOT.TFile("Higgs_data.root")

inputFile.ls()

#inputFile.sig_tree.Print()

TMVA.Tools.Instance();

#optional output file
outputFile = ROOT.TFile("TMVAOutput.root", "RECREATE")

factory = TMVA.Factory( "TMVAClassification", outputFile #this is optional
                       ,"!V:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification" )

factory = TMVA.Factory("TMVAClassification", TargetFile=outputFile,
                       V=False, Color=True, DrawProgressBar=True, Transformations=["I", "D", "P", "G", "D"],
                       AnalysisType="Classification")

loader = TMVA.DataLoader("dataset");

#add variables (we use the high level ones)
loader.AddVariable("m_jj")
loader.AddVariable("m_jjj")
loader.AddVariable("m_lv")
loader.AddVariable("m_jlv")
loader.AddVariable("m_bb")
loader.AddVariable("m_wbb")
loader.AddVariable("m_wwbb")

mycuts = ROOT.TCut()
mycutb = ROOT.TCut()


loader.AddSignalTree    (inputFile.sig_tree,     1.0)   #signal weight  = 1
loader.AddBackgroundTree(inputFile.bkg_tree, 1.0);   #background weight = 1 

 
loader.PrepareTrainingAndTestTree(SigCut=mycuts, BkgCut=mycutb,
                    nTrain_Signal=1000, nTrain_Background=1000, nTest_Signal=10000, nTest_Background=10000, SplitMode="Random", NormMode="NumEvents", V=False)

loader.DrawInputVariable("m_wwbb")

loader.DrawInputVariable("m_jj")

loader.DrawCorrelationMatrix("Signal")

#factory.BookMethod( DataLoader=loader, Method=TMVA.Types.kSVM, MethodTitle="SVM", 
#                Gamma=0.25, Tol=0.001, VarTransform="Norm" )


factory.BookMethod( loader,TMVA.Types.kLD, "LD", 
    H=False, V=False, VarTransform="None", CreateMVAPdfs=True, PDFInterpolMVAPdf="Spline2",
                   NbinsMVAPdf=50, NsmoothMVAPdf=10 )


factory.BookMethod( loader, TMVA.Types.kBDT, "BDT",
    H=False, V=False, NTrees=850, MinNodeSize="2.5%", MaxDepth=3, BoostType="AdaBoost", AdaBoostBeta=0.5,
                   UseBaggedBoost=True, BaggedSampleFraction=0.5, SeparationType="GiniIndex", nCuts=20 )

bookDNN = False
DNNType = 'Standard'
#DNNType = 'CPU'   # use new implementation 
trainingStrategyFast = [{
        "LearningRate": 1e-1,
        "Momentum": 0.0,
        "Repetitions": 1,
        "ConvergenceSteps": 100,
        "BatchSize": 50,
        "TestRepetitions": 7,
        "WeightDecay": 0.001,
        "Regularization": "NONE",
        "DropConfig": "0.0+0.5+0.5+0.5",
        "DropRepetitions": 1,
        "Multithreading": True
        
    }]
trainingStrategy = [{
        "LearningRate": 1e-1,
        "Momentum": 0.0,
        "Repetitions": 1,
        "ConvergenceSteps": 100,
        "BatchSize": 50,
        "TestRepetitions": 4,
        "WeightDecay": 0.001,
        "Regularization": "NONE",
        "DropConfig": "0.0+0.5+0.5+0.5",
        "DropRepetitions": 1,
        "Multithreading": True
        
    },  {
        "LearningRate": 1e-2,
        "Momentum": 0.5,
        "Repetitions": 1,
        "ConvergenceSteps": 100,
        "BatchSize": 50,
        "TestRepetitions": 2,
        "WeightDecay": 0.001,
        "Regularization": "TRUE",
        "DropConfig": "0.0+0.1+0.1+0.1",
        "DropRepetitions": 1,
        "Multithreading": True
        
    }, {
        "LearningRate": 1e-3,
        "Momentum": 0.3,
        "Repetitions": 1,
        "ConvergenceSteps": 100,
        "BatchSize": 50,
        "TestRepetitions": 2,
        "WeightDecay": 0.001,
        "Regularization": "NONE",
        "Multithreading": True
    
        
}]

if (bookDNN) : 
    factory.BookMethod(DataLoader=loader, Method=TMVA.Types.kDNN, MethodTitle="DNN", 
                   H = False, V=False, VarTransform="Normalize", ErrorStrategy="CROSSENTROPY",
                   Layout=["TANH|50", "TANH|50", "TANH|10", "LINEAR"],
                   TrainingStrategy=trainingStrategyFast,Architecture=DNNType)

factory.TrainAllMethods();

factory.TestAllMethods();
factory.EvaluateAllMethods();    

factory.DrawOutputDistribution(loader.GetName(), "LD")

factory.DrawOutputDistribution(loader.GetName(), "BDT")

factory.DrawOutputDistribution(loader.GetName(), "Fisher")

if (bookDNN) : factory.DrawOutputDistribution(loader.GetName(), "DNN")

roc = factory.GetROCCurve(loader)
roc.Draw();

print 'ROC integral for LD = ',factory.GetROCIntegral(loader, 'LD')

print 'ROC integral for BDT = ',factory.GetROCIntegral(loader, 'BDT')

if (bookDNN) : print 'ROC integral for DNN = ',factory.GetROCIntegral(loader, 'DNN')

if (bookDNN):  factory.DrawNeuralNetwork(loader.GetName(), "DNN")

