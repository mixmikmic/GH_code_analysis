# We start by importing all the necessary packages
import sys;
import os, sys, array, re, math, random, subprocess, glob
from math import *
import numpy as np
import scipy
from numpy.lib.recfunctions import stack_arrays
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import cPickle
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils import compute_class_weight
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Highway, MaxoutDense, Masking, GRU, Merge, Input, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
import deepdish.io as io
import ROOT
from ROOT import gSystem, gROOT, gApplication, TFile, TTree, TCut, TH1F, TCanvas
from root_numpy import root2array 
from IPython.display import HTML, IFrame
import seaborn as sns; sns.set()
print "If you had no error so far, this is great! We can start the tutorial."

print "I'm now importing an external file with few function and defining the input/output files location."
# Check this file out, it contains many functions we will use.
execfile("Useful_func.py")
# Fix random seed for reproducibility
seed = 7; np.random.seed(seed);
# Input paramters
debug = True #(Verbose output)
folder='Plots_hh_tt_MLP/' # Folder with Plots
MakePlots=True # Set False if you want to run faster
folderCreation  = subprocess.Popen(['mkdir -p ' + folder], stdout=subprocess.PIPE, shell=True); folderCreation.communicate()
folderCreation2 = subprocess.Popen(['mkdir -p models/'], stdout=subprocess.PIPE, shell=True); folderCreation2.communicate()
# Sample location
h_file = "files/MVA_GluGluToRadionToHHTo2B2VTo2L2Nu_M-500_narrow_13TeV-madgraph-v2.root"
TT_df_list = ['files/tt_dataframe_0.csv','files/tt_dataframe_1.csv','files/tt_dataframe_2.csv','files/tt_dataframe_3.csv','files/tt_dataframe_4.csv','files/tt_dataframe_5.csv']

print "Now we manipulate the input ROOT files into a format we can pass to Keras."
# Our goal is to separate Signal from Background. We need to select only "good events".
# 1) We apply a preselection to our Signal and Background events
my_selec = 'met_pt>20 && met_pt<500 && muon1_pt>20 && fabs(muon1_eta)<2.4 && muon2_pt>10 && fabs(muon2_eta)<2.4 && pt_l1l2<500 && pt_b1b2<500 && mass_l1l2>12 && mass_l1l2<500 && mass_b1b2<500 && b1jet_pt>20 && fabs(b1jet_eta)<2.4 && b2jet_pt>20 && fabs(b2jet_eta)<2.4 && mass_trans>10 && mass_trans<500 && HT<4000'
# 2) We select the TTree branches that contains the information we want to keep
my_branches = ["MT2","mass_trans","dphi_llmet","dphi_llbb","eta_l1l2","pt_l1l2","mass_l1l2","eta_b1b2","pt_b1b2","mass_b1b2","dR_minbl","dR_l1l2b1b2","HT","met_pt","muon1_pogSF","muon2_pogSF","XsecBr"]
# 3) We select the TTree branches that contains the information we want to use in the training (FEATURES)
my_branches_training = ["MT2","mass_trans","dphi_llmet","dphi_llbb","eta_l1l2","pt_l1l2","mass_l1l2","eta_b1b2","pt_b1b2","mass_b1b2","dR_minbl","dR_l1l2b1b2","HT","met_pt"]
    
# Converting Root files into a DATAFRAME (Very useful, checnl root2panda in Useful_func.py)
hh    = root2panda(h_file, 'DiHiggsWWBBAna/evtree', branches=my_branches, selection=my_selec)
## TT is so heavy that I already saved the final dataframe (and I splitted it if 5 so that each df is smaller than 100 Mb).
ttbar_df_list = (pd.read_csv(f_df) for f_df in TT_df_list)
ttbar = pd.concat(ttbar_df_list, ignore_index=True)
ttbar = ttbar.drop(ttbar.columns[[0]], 1) #Remove 2 extra columns we have in this df, so it match with Signal

# These processes have a different cross-sextion. You need to know how to weight each event such that yuo can estimate the real number of events expected in a given Luminosity.
# The weight is (Xsec*Br)*Lumi/N_totalEvent_genrated.
# First you need to knwow the Total number of MC events generated
MyFile_hh =  ROOT.TFile.Open(h_file,"read");
h_prehlt_hh = ROOT.TH1F(MyFile_hh.Get("TriggerResults/hevent_filter"))
nTOT_prehlt_hh = h_prehlt_hh.GetBinContent(2) # Here is stored the number of total event generated
nTOT_prehlt_ttbar = 102114184 #Harcoded
Lumi = 36.42 * 1000 #1000 is for passing from fb-1 to pb-1 (in which the xsec is expressed). 
hh['XsecBr'] = hh['XsecBr']*Lumi/nTOT_prehlt_hh
ttbar['XsecBr'] = ttbar['XsecBr']*Lumi/nTOT_prehlt_ttbar
# Add all weights in the df (you are adding the Data/MC scale factors too)
hh['fin_weight']    = hh['XsecBr'] * hh['muon1_pogSF'] * hh['muon2_pogSF'] #1pb is S Xsec.
ttbar['fin_weight'] = ttbar['XsecBr'] * ttbar['muon1_pogSF'] * ttbar['muon2_pogSF'] #87pb is TT Xsec.
print "Assuming a Signal of 1pb and a background of 87pb (B.R. included) you expect, after the preselection, to have:",hh['fin_weight'].sum(),"S and ",ttbar['fin_weight'].sum(),"B"

## Alternatively you can save a df as a h5 file (for quick loading in the future)
#  Ex: io.save(open('models/ttbar.h5', 'wb'), ttbar); ttbar = io.load(open('models/ttbar.h5', 'rb'));
if debug:
    print("---> hh Displayed as panda dataframe: "); print(hh)
    print("The shape for hh is (samples, features): "); print(hh.shape)
    print("The shape for tt is (samples, features): "); print(ttbar.shape)
    print hh.keys()
    print ttbar.keys()

get_ipython().magic('matplotlib inline')
# Plots of the branches we selected
if MakePlots:
    print "Producing plots of the features in S and B."
    for key in ttbar.keys() :
        if(key!="muon1_pogSF" and key!="muon2_pogSF" and key!="XsecBr" and key!="fin_weight") :
            matplotlib.rcParams.update({'font.size': 16})
            fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
            bins = np.linspace(my_max(min(ttbar[key]),0.), max(ttbar[key]), 50)
            _ = plt.hist(hh[key],  bins=bins, histtype='step', normed=True, label=r'$hh$', linewidth=2)
            _ = plt.hist(ttbar[key], bins=bins, histtype='step', normed=True, label=r'$t\overline{t}$')
            plt.xlabel(key)
            plt.ylabel('Entries')
            plt.legend(loc='best')
            plt.savefig(folder + "/" + str(key) + '.pdf')
        

# Lets look at the correlations of the features
c1 = ROOT.TCanvas(); c1.cd(); ROOT.gStyle.SetOptStat(0)
if MakePlots:
    print "Plotting the correlation of the features in S and B."
    h_Corr_hh    = ROOT.TH2F("h_Corr_hh","", len(my_branches_training), 0, len(my_branches_training), len(my_branches_training), 0, len(my_branches_training))
    h_Corr_ttbar = ROOT.TH2F("h_Corr_ttbar","", len(my_branches_training), 0, len(my_branches_training), len(my_branches_training), 0, len(my_branches_training))
    for var1 in range(len(my_branches_training)):
        h_Corr_hh.GetXaxis().SetBinLabel(var1+1,my_branches_training[var1])
        h_Corr_ttbar.GetXaxis().SetBinLabel(var1+1,my_branches_training[var1])
        for var2 in range(len(my_branches_training)):
            h_Corr_hh.GetYaxis().SetBinLabel(var2+1,my_branches_training[var2])
            h_Corr_ttbar.GetYaxis().SetBinLabel(var2+1,my_branches_training[var2])
            if(var2>=var1):
                array_Var1_hh_var1    = np.array( hh[my_branches_training[var1]] )
                array_Var1_hh_var2    = np.array( hh[my_branches_training[var2]] )
                array_Var1_ttbar_var1 = np.array( ttbar[my_branches_training[var1]] )
                array_Var1_ttbar_var2 = np.array( ttbar[my_branches_training[var2]] )
                corr = scipy.stats.pearsonr( array_Var1_hh_var1, array_Var1_hh_var2 )[0]
                h_Corr_hh.SetBinContent(var1+1,var2+1,corr)
                corr = scipy.stats.pearsonr( array_Var1_ttbar_var1, array_Var1_ttbar_var2 )[0]
                h_Corr_ttbar.SetBinContent(var1+1,var2+1,corr)
    h_Corr_hh.GetZaxis().SetRangeUser(-1.,1.)
    h_Corr_ttbar.GetZaxis().SetRangeUser(-1.,1.)
    ROOT.gStyle.SetPaintTextFormat(".2f");
    h_Corr_hh.Draw("colzTEXT")
    c1.SaveAs(folder + '/Corr_hh.pdf')
    h_Corr_ttbar.Draw("colzTEXT")
    c1.SaveAs(folder + '/Corr_ttbar.pdf')
    
    

print('Now lets start to talk about DNN!')
#You only need a Dataframe for the training. So you merge all the one you have
df =  pd.concat((hh[my_branches_training], ttbar[my_branches_training]), ignore_index=True)
# Turn the df the desired ndarray "X" that can be directly used for ML applications.
X = df.as_matrix() # Each row is an object to classify, each column corresponds to a specific variable.
# Take the weights
w =  pd.concat((hh['fin_weight'], ttbar['fin_weight']), ignore_index=True).values
# This is the array with the true values: 0 is signal, 1 if TT.
y = []
for _df, ID in [(hh, 0), (ttbar, 1)]:
    y.extend([ID] * _df.shape[0]) # You give to y the ID value (first 0, then 1) a number of time equal to the raw in hh and ttbar
print "y is"
y = np.array(y) # Simply a conversion tu numpy, but still a vector of 0 and 1 (for S and B)

# Randomly shuffle and automatically split all your objects into train and test subsets
ix = range(X.shape[0]) # array of indices, just to keep track of them for safety reasons and future checks
X_train, X_test, y_train, y_test, w_train, w_test, ix_train, ix_test = train_test_split(X, y, w, ix, train_size=0.7) # Train here is 70% of the total statistic
# It is common practice to scale the inputs to Neural Nets such that they have approximately similar ranges (it atually improve the results)
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) # You are applying the same transformation done to X_train, to X_test.

# This takes a while, but it is worth do do it once. It shows the correlations for S and B overimposed.
# More variables you add in var_toPlot, more times it takes.
if MakePlots:
    # Next lines to select the same number of entries
    rawS = hh.shape[0]
    rawB = ttbar.shape[0]
    Nraw = rawS;
    if (rawB<rawS): Nraw = rawB;
    hh_plot    = hh.iloc[0:Nraw]
    ttbar_plot = ttbar.iloc[0:Nraw]
    # Add the target, that is missing in hh and ttbar
    hh_plot['Target'] = 0
    ttbar_plot['Target'] = 1
    var_toPlot=["Target","dphi_llmet","pt_l1l2","mass_l1l2","pt_b1b2","mass_b1b2","HT","met_pt"]
    df_withY = pd.concat((hh_plot[var_toPlot], ttbar_plot[var_toPlot]), ignore_index=True)
    # You can select the variable to plot in sns.pairplot using an argumnet vars=['var1','var2'...]
    sns_plot = sns.pairplot(df_withY, hue='Target',palette=["#e74c3c","#9b59b6"],plot_kws={"s": 3,"alpha":0.3},size=5)
    sns_plot.savefig(folder + "/Variables_pairplot.png")

# Multilayer Perceptron (MLP) definition
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu')) # Linear transformation of the input vector. The first number is output_dim.
model.add(Dropout(0.1)) # To avoid overfitting. It masks the outputs of the previous layer such that some of them will randomly become inactive and will not contribute to information propagation.
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(20, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(10,activation='tanh'))
model.add(Dropout(0.1))
model.add(Dense(20, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(2, activation='softmax')) # Last layer has to have the same dimensionality as the number of classes we want to predict, here 2.
model.summary()
# Now you need to declare what loss function and optimizer to use (and compile your model).
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Metrics are evaluated by the model during training (here accuracy)
# The loss function, also called the objective function is the evaluation of the model used by the optimizer to navigate the weight space.
#‘mse‘: for mean squared error.
#‘binary_crossentropy‘: for binary logarithmic loss (logloss).
#‘categorical_crossentropy‘: for multi-class logarithmic loss (logloss).
# http://keras.io/objectives/

print('---------------------------Training:---------------------------')
try:
    history = model.fit(X_train, y_train, batch_size=50, epochs=100, verbose=1,
              callbacks = [
                  EarlyStopping(verbose=True, patience=6, monitor='val_loss'),
                  ModelCheckpoint('models/tutorial-progress.h5', monitor='val_loss', verbose=1, save_best_only=True)
              ],
              validation_split=0.2, validation_data=None, shuffle=True,
              class_weight={
                0 : compute_class_weight("balanced", [0, 1], y)[0], # Function that return "[1/N_classes * ((float(len(y)) / (y == 0).sum())), 1/N_classes * ((float(len(y)) / (y == 1).sum()))]"
                1 : compute_class_weight("balanced", [0, 1], y)[1]
              },
              sample_weight=None,initial_epoch=0)
    
except KeyboardInterrupt:
    print 'Training ended early.'

# Epochs is the number of times that the model is exposed to the training dataset.
# Batch Size is the number of training instances shown to the model before a weight update is performed.

# Let's make some plots about the convergence of our model. You can check if:
#  1. It’s speed of convergence over epochs (slope).
#  2. Whether the model may have already converged (plateau of the line).
#  3. Whether the mode may be over-learning the training data (inflection for validation line).

if debug:
    print "All data in history are: ",history.history.keys()
# Summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig(folder + '/Check_accuracy.pdf')
# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig(folder + '/Check_loss.pdf')

# Load the best network (by default you return the last one, you if you save every time you have a better one you are fine loading it later)
model.load_weights('./models/tutorial-progress.h5')
print 'Saving weights...'
model.save_weights('./models/tutorial.h5', overwrite=True)
json_string = model.to_json()
open('./models/tutorial.json', 'w').write(json_string)
print 'Testing...'
yhat = model.predict(X_test, verbose = True, batch_size = 50) # Return a vector of 2 indexes [probToBe_S,probToBe_B]
#Turn them into classes
yhat_cls = np.argmax(yhat, axis=1) # Transform [probToBe_S,probToBe_B] in a vector of 0 and 1 depending if probToBe_S>probToBe_B. Practically return the index of the biggest element (0 is is probToBe_S, if is probToBe_B)
#model.evaluate(): To calculate the loss values for input data.
#model.predict(): To generate network output for input data.
#model.predict_classes(): To generate class outputs for input data.
#model.predict_proba(): To generate class probabilities for input data.

if debug:
    print; print "y_test is an array:",y_test
    print "yhat is an array of [probToBe_S,probToBe_B]: ",yhat
    print "yhat_cls is an array:",yhat_cls

# A Plot of the tructure of the MLP (to see the details of your DNN)
from keras.utils import plot_model
plot_model(model, to_file=folder+'/model.png')

if MakePlots:
    # Check the predicted and true S and B (Normalized to the expected events for the given Lumi)
    bins = np.linspace(-0.5,1.5,3)
    names = ['','','','hh','','','','tt']
    fig = plt.figure(figsize=(7.69, 5.27), dpi=100)
    ax = plt.subplot()
    ax.set_xticklabels(names, rotation=45)
    _ = plt.hist(yhat_cls, bins=bins, histtype='step', color='red', label='Prediction',log=True, weights=w_test)
    _ = plt.hist(y_test, bins=bins, histtype='stepfilled', alpha=0.4, color='blue',label='Truth',log=True, weights=w_test)
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig(folder + '/Performance.pdf')
    # Check the score (y_test==0 is a vector of bools). Ideally you want a distribution that is all red on left and all blue on right.
    fig = plt.figure(figsize=(7.69, 5.27), dpi=100)
    plt.hist(yhat_cls[y_test==0], label='Signal', normed=True, histtype='step', color='red', weights=w_test[y_test==0])
    plt.hist(yhat_cls[y_test==1], label='Background', normed=True, histtype='step', color='blue', weights=w_test[y_test==1])
    plt.legend()
    plt.show()
    plt.savefig(folder + '/Performance2.pdf')
    # Check it on the training (should be similar to the test)
    yhat_train = model.predict(X_train, verbose = True, batch_size = 50)
    yhat_cls_train = np.argmax(yhat_train, axis=1)
    fig = plt.figure(figsize=(7.69, 5.27), dpi=100)
    plt.hist(yhat_cls_train[y_train == 0], label='Signal train', normed=True, histtype='step', color='red', weights=w_train[y_train==0])
    plt.hist(yhat_cls_train[y_train == 1], label='Background train', normed=True, histtype='step', color='blue', weights=w_train[y_train==1])
    plt.legend()
    plt.show()
    plt.savefig(folder + '/Performance2_train.pdf')

# With "(y_test != 0) & (yhat_cls == 0)" you get an arrate of bool: [False False False ..., False False False]
print 'Signal efficiency:',     w_test[(y_test == 0) & (yhat_cls == 0)].sum() / w_test[y_test == 0].sum(),"(",w_test[(y_test == 0) & (yhat_cls == 0)].sum(),"/",w_test[y_test == 0].sum(),")"
print 'Background efficiency:', w_test[(y_test != 0) & (yhat_cls == 0)].sum() / w_test[y_test != 0].sum(),"(",w_test[(y_test != 0) & (yhat_cls == 0)].sum(),"/",w_test[y_test != 0].sum(),")"
print "S/sqrt(S+B) = ", (w_test[(y_test == 0) & (yhat_cls == 0)].sum())/sqrt(w_test[(y_test == 0) & (yhat_cls == 0)].sum()+w_test[(y_test != 0) & (yhat_cls == 0)].sum())
w_1 = np.ones(len(y_test))
print 'Signal efficiency (not weighted):',     w_1[(y_test == 0) & (yhat_cls == 0)].sum() / w_1[y_test == 0].sum()
print 'Background efficiency (not weighted):', w_1[(y_test != 0) & (yhat_cls == 0)].sum() / w_1[y_test != 0].sum()
print "Let's compare with the training samples:"
w_1 = np.ones(len(y_train))
yhat_tr = model.predict(X_train, verbose = True, batch_size = 50)
yhat_trcls = np.argmax(yhat_tr, axis=1)
print ''; print 'Signal efficiency (not weighted) for training:',     w_1[(y_train == 0) & (yhat_trcls == 0)].sum() / w_1[y_train == 0].sum()
print 'Background efficiency (not weighted) for training:', w_1[(y_train != 0) & (yhat_trcls == 0)].sum() / w_1[y_train != 0].sum()

# Let's start with low mass
hh3     = root2panda('files/MVA_GluGluToRadionToHHTo2B2VTo2L2Nu_M-300_narrow_13TeV-madgraph-v2.root', 'DiHiggsWWBBAna/evtree', branches=my_branches, selection=my_selec)
hh3['fin_weight']    = hh3['XsecBr'] * hh3['muon1_pogSF'] * hh3['muon2_pogSF'] #1pb
df3 =  pd.concat((hh3[my_branches_training], ttbar[my_branches_training]), ignore_index=True)
# X, w, Y
X3 = df3.as_matrix() # Each row is an object to classify, each column corresponds to a specific variable.
w3 =  pd.concat((hh3['fin_weight'], ttbar['fin_weight']), ignore_index=True).values
y3 = []
for _df, ID in [(hh3, 0), (ttbar, 1)]:
    y3.extend([ID] * _df.shape[0])
y3 = np.array(y3)
# Scaler
X3 = scaler.transform(X3)
yhat3 = model.predict(X3, verbose = True, batch_size = 50) # Return a vector of 2 indeces [probToBe_S,probToBe_B]
#Turn them into classes
yhat3_cls = np.argmax(yhat3, axis=1) 
# And Now the efficiency:
print; print 'Signal efficiency:',     w3[(y3 == 0) & (yhat3_cls == 0)].sum() / w3[y3 == 0].sum(), "(",w3[(y3 == 0) & (yhat3_cls == 0)].sum(),"/",w3[y3 == 0].sum(),")"
print 'Background efficiency:', w3[(y3 != 0) & (yhat3_cls == 0)].sum() / w3[y3 != 0].sum(), "(",w3[(y3 != 0) & (yhat3_cls == 0)].sum(),"/",w3[y3 != 0].sum(),")"
print "S/sqrt(S+B) = ", w3[(y3 == 0) & (yhat3_cls == 0)].sum()/sqrt(w3[(y3 == 0) & (yhat3_cls == 0)].sum()+w3[(y3 != 0) & (yhat3_cls == 0)].sum())

# And now high mass
hh13    = root2panda('files/MVA_GluGluToRadionToHHTo2B2VTo2L2Nu_M-900_narrow_13TeV-madgraph-v2.root', 'DiHiggsWWBBAna/evtree', branches=my_branches, selection=my_selec)
hh13['fin_weight']    = hh13['XsecBr'] * hh13['muon1_pogSF'] * hh13['muon2_pogSF'] #1pb
df13 =  pd.concat((hh13[my_branches_training], ttbar[my_branches_training]), ignore_index=True)
# X, w, Y
X13 = df13.as_matrix() # Each row is an object to classify, each column corresponds to a specific variable.
w13 =  pd.concat((hh13['fin_weight'], ttbar['fin_weight']), ignore_index=True).values
y13 = []
for _df, ID in [(hh13, 0), (ttbar, 1)]:
    y13.extend([ID] * _df.shape[0])
y13 = np.array(y13)
# Scaler
X13 = scaler.transform(X13)
yhat13 = model.predict(X13, verbose = True, batch_size = 50) # Return a vector of 2 indeces [probToBe_S,probToBe_B]
#Turn them into classes
yhat13_cls = np.argmax(yhat13, axis=1) 
# And Now the efficiency:
print; print 'Signal efficiency:',     w13[(y13 == 0) & (yhat13_cls == 0)].sum() / w13[y13 == 0].sum(),"(",w13[(y13 == 0) & (yhat13_cls == 0)].sum(),"/",w13[y13 == 0].sum(),")"
print 'Background efficiency:', w13[(y13 != 0) & (yhat13_cls == 0)].sum() / w13[y13 != 0].sum(),"(",w13[(y13 != 0) & (yhat13_cls == 0)].sum(),"/",w13[y13 != 0].sum(),")"
print "S/sqrt(S+B) = ",w13[(y13 == 0) & (yhat13_cls == 0)].sum() /sqrt(w13[(y13 == 0) & (yhat13_cls == 0)].sum() + w13[(y13 != 0) & (yhat13_cls == 0)].sum())

# Let's run a MLP on a low mass Signal
# Let's start with low mass
hh3N     = root2panda('files/MVA_GluGluToRadionToHHTo2B2VTo2L2Nu_M-300_narrow_13TeV-madgraph-v2.root', 'DiHiggsWWBBAna/evtree', branches=my_branches, selection=my_selec)
hh3N['fin_weight'] = hh3N['XsecBr'] * hh3N['muon1_pogSF'] * hh3N['muon2_pogSF'] #1pb
df3N =  pd.concat((hh3N[my_branches_training], ttbar[my_branches_training]), ignore_index=True)
# X, w, Y
X3N = df3N.as_matrix() # Each row is an object to classify, each column corresponds to a specific variable.
w3N =  pd.concat((hh3N['fin_weight'], ttbar['fin_weight']), ignore_index=True).values
y3N = []
for _df, ID in [(hh3N, 0), (ttbar, 1)]:
    y3N.extend([ID] * _df.shape[0])
y3N = np.array(y3)
# Randomly shuffle and automatically split all your objects into train and test subsets
ix3N = range(X3N.shape[0]) # array of indices, just to keep track of them for safety reasons and future checks
X3N_train, X3N_test, y3N_train, y3N_test, w3N_train, w3N_test, ix3N_train, ix3N_test = train_test_split(X3N, y3N, w3N, ix3N, train_size=0.7) # Train here is 70% of the total statistic
# It is common practice to scale the inputs to Neural Nets such that they have approximately similar ranges (it atually improve the results)
scaler3N = StandardScaler() 
X3N_train = scaler3N.fit_transform(X3N_train)
X3N_test = scaler3N.transform(X3N_test) # You are applying the same transformation done to X_train, to X_test.


print('---------------------------Training:---------------------------')
try:
    history3N = model.fit(X3N_train, y3N_train, batch_size=50, epochs=100, verbose=1,
              callbacks = [
                  EarlyStopping(verbose=True, patience=6, monitor='val_loss'),
                  ModelCheckpoint('models/tutorial-progress3N.h5', monitor='val_loss', verbose=1, save_best_only=True)
              ],
              validation_split=0.2, validation_data=None, shuffle=True,
              class_weight={
                0 : compute_class_weight("balanced", [0, 1], y3N)[0], # Function that return "[1/N_classes * ((float(len(y)) / (y == 0).sum())), 1/N_classes * ((float(len(y)) / (y == 1).sum()))]"
                1 : compute_class_weight("balanced", [0, 1], y3N)[1]
              },
              sample_weight=None,initial_epoch=0)
    
except KeyboardInterrupt:
    print 'Training ended early.'

if debug:
    print "All data in hisotry are: ",history3N.history.keys()
# Summarize history for accuracy
plt.plot(history3N.history['acc'])
plt.plot(history3N.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig(folder + '/Check_accuracy_300GeV.pdf')
# Summarize history for loss
plt.plot(history3N.history['loss'])
plt.plot(history3N.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig(folder + '/Check_loss_300GeV.pdf')

# Load the best network (by default you return the last one, you if you save every time you have a better one you are fine loading it later)
model.load_weights('./models/tutorial-progress3N.h5')
print 'Saving weights...'
model.save_weights('./models/tutorial3N.h5', overwrite=True)
json_string = model.to_json()
open('./models/tutorial3N.json', 'w').write(json_string)
print 'Testing...'
yhat3N = model.predict(X3N_test, verbose = True, batch_size = 50) # Return a vector of 2 indeces [probToBe_S,probToBe_B]
#Turn them into classes
yhat3N_cls = np.argmax(yhat3N, axis=1)

#Now the efficiency
print; print 'Signal efficiency (not weighted):',     w3N[(y3N_test == 0) & (yhat3N_cls == 0)].sum() / w3N[y3N_test == 0].sum(),"(",w3N[(y3N_test == 0) & (yhat3N_cls == 0)].sum(),"/",w3N[y3N_test == 0].sum(),")"
print 'Background efficiency (not weighted):', w3N[(y3N_test != 0) & (yhat3N_cls == 0)].sum() / w3N[y3N_test != 0].sum(),"(",w3N[(y3N_test != 0) & (yhat3N_cls == 0)].sum(),"/",w3N[y3N_test != 0].sum(),")"
print "S/sqrt(S+B) = ", w3N[(y3N_test == 0) & (yhat3N_cls == 0)].sum()/sqrt(w3N[(y3N_test == 0) & (yhat3N_cls == 0)].sum()+w3N[(y3N_test != 0) & (yhat3N_cls == 0)].sum())

# Let's start with low mass
hh5     = root2panda('files/MVA_GluGluToRadionToHHTo2B2VTo2L2Nu_M-500_narrow_13TeV-madgraph-v2.root', 'DiHiggsWWBBAna/evtree', branches=my_branches, selection=my_selec)
hh5['fin_weight']    = hh5['XsecBr'] * hh5['muon1_pogSF'] * hh5['muon2_pogSF'] #1pb
df5 =  pd.concat((hh5[my_branches_training], ttbar[my_branches_training]), ignore_index=True)
# X, w, Y
X5 = df5.as_matrix() # Each row is an object to classify, each column corresponds to a specific variable.
w5 = pd.concat((hh5['fin_weight'], ttbar['fin_weight']), ignore_index=True).values
y5 = []
for _df, ID in [(hh5, 0), (ttbar, 1)]:
    y5.extend([ID] * _df.shape[0])
y5 = np.array(y5)
# Scaler
X5 = scaler3N.transform(X5)
yhat5 = model.predict(X5, verbose = True, batch_size = 50) # Return a vector of 2 indeces [probToBe_S,probToBe_B]
#Turn them into classes
yhat5_cls = np.argmax(yhat5, axis=1) 
# And Now the efficiency:
print; print 'Signal efficiency:',     w5[(y5 == 0) & (yhat5_cls == 0)].sum() / w5[y5 == 0].sum(),"(",w5[(y5 == 0) & (yhat5_cls == 0)].sum(),"/",w5[y5 == 0].sum(),")"
print 'Background efficiency:', w5[(y5 != 0) & (yhat5_cls == 0)].sum() / w5[y5 != 0].sum(),"(",w5[(y5 != 0) & (yhat5_cls == 0)].sum(),"/", w5[y5 != 0].sum(),")"
print "S/sqrt(S+B) = ", w5[(y5 == 0) & (yhat5_cls == 0)].sum()/sqrt(w5[(y5 == 0) & (yhat5_cls == 0)].sum() + w5[(y5 != 0) & (yhat5_cls == 0)].sum())

