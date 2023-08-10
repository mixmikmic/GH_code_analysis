get_ipython().magic('matplotlib inline')
if __package__ is None:
    import sys, os
    sys.path.append(os.path.realpath("/data/shared/Software/"))
from CMS_SURF_2016.utils.archiving import *
from CMS_SURF_2016.layers.slice import Slice
from CMS_SURF_2016.layers.lorentz import Lorentz
from keras.utils.visualize_util import plot
from IPython.display import Image, display
from CMS_SURF_2016.utils.colors import colors_contrasting
from CMS_SURF_2016.utils.analysistools import *
from CMS_SURF_2016.utils.plot import *
from CMS_SURF_2016.utils.metrics import *
import numpy as np
from CMS_SURF_2016.utils.plot import plot_history, print_accuracy_m
archive_dir = "/data/shared/Delphes/CSCS_output/keras_archive/"


#def sortTrialsOn(lst):
    

trials = get_trials_by_name("(not_lorentz|lorentz|control_dense)", archive_dir)
#Verbose changes the hashing, which is a pending bug, the trials with verbose = 0
#are the newest oned built by LorentzVsDenseTrials2.py
trials = findWithMetrics(trials, {"verbose": 0})
for trial in trials:
    trial.summary(showTraining=False,showValidation=False, showFit=False, showCompilation=True)
print("TotalNumber of Trials:", len(trials)) 



print_by_labels(trials,num_print=2,sortMetric='val_acc')

three_ways = findWithMetrics(trials, {"labels" : [u'ttbar', u'wjet', u'qcd']})
sortOnMetric(three_ways)
plotEverything(three_ways[:4], custom_objects={"Slice":Slice, "Lorentz" : Lorentz})

control = findWithMetrics(get_trials_by_name("not_lorentz", archive_dir), {"verbose": 0})
control_dense = findWithMetrics(get_trials_by_name("control_dense", archive_dir), {"verbose": 0})
lorentz = findWithMetrics(get_trials_by_name("lorentz", archive_dir), {"verbose": 0})
#control = findWithMetrics(trials, {"name" : "not_lorentz"})
#lorentz = findWithMetrics(trials, {"name" : "lorentz"})
sortOnMetric(control, "val_acc")
sortOnMetric(control_dense, "val_acc")
sortOnMetric(lorentz, "val_acc")
labels = ['ttbar', 'wjet', 'qcd']
#print_by_labels(trials, 4)
best_controls = control[:4]
best_control_denses = control_dense[:4]
best_lorentzs = lorentz[:4]

plotEverything(best_controls, custom_objects={"Slice":Slice, "Lorentz" : Lorentz})

plotEverything(best_control_denses, custom_objects={"Slice":Slice, "Lorentz" : Lorentz})

plotEverything(best_lorentzs, custom_objects={"Slice":Slice, "Lorentz" : Lorentz})

plotMetricVsMetric(control, "depth", metricY="val_acc", groupOn="labels",
                       ylabel="Validation Accuracy",legend_label="labels", constants={"weight_output" : False, "width" : 10}, mode="error", alpha=.7, verbose=0)
plotMetricVsMetric(control_dense, "depth", metricY="val_acc", groupOn="labels",
                       ylabel="Validation Accuracy",legend_label="labels", constants={"weight_output" : False, "width" : 10}, mode="error", alpha=.7, verbose=0)
plotMetricVsMetric(lorentz, "depth", metricY="val_acc", groupOn="labels",
                       ylabel="Validation Accuracy",legend_label="labels", constants={"weight_output" : False, "width" : 10}, mode="error", alpha=.7, verbose=0)

top_lorentz = findWithMetrics(lorentz, {"depth" : 5,
                                        "width": 10,
                                       "weight_output": False})
plotEverything(top_lorentz)

top_lorentz_weighted = findWithMetrics(lorentz, {"depth" : 5,
                                        "width": 10,
                                       "weight_output": True})
plotEverything(top_lorentz_weighted)

matching_control = findWithMetrics(control, {"depth" : 5,
                                        "width": 10})
plotEverything(matching_control)

matching_control_dense = findWithMetrics(control_dense, {"depth" : 5,
                                        "width": 10})
plotEverything(matching_control_dense)

sortOnMetric(top_lorentz, "labels")
sortOnMetric(top_lorentz_weighted, "labels")
sortOnMetric(matching_control, "labels")
sortOnMetric(matching_control_dense, "labels")
labelGroups = zip(top_lorentz,top_lorentz_weighted,matching_control,matching_control_dense)

colors = [(0,0,1.0),(.25,.75,.25), (1,0,0), (1,.65,0)]
names = ["Lorentz", "Lorentz_w","Ctrl", "Ctrl_d"]
lims = [[.815,.822], [.93,.97], [.97,.99], [.715, .75]]


for j , tup in enumerate(labelGroups):
    plots = []
    for i, b in enumerate(tup):
        labels = b.get_from_record("labels")
        if(labels == None): labels = b.get_from_record("lables")
        title = str(tuple([str(x) for x in labels])) if(labels != None) else "Cannot Find Labels"
        title = title + " Accuracy vs Epoch"
        name = names[i]
        model = b.get_model(custom_objects={"Slice":Slice, "Lorentz" : Lorentz})
        history = b.get_history()
        color = colors[i]
        plots.append((name, history, color))
    plot_history(plots, plotLoss=False, title=title, acclims=lims[j])


    

maxSizeDict = {}
for trial in top_lorentz:
    labels = tuple(trial.get_from_record("labels"))
    maxSizeDict[labels] = {}
    p = DataProcedure.from_json(archive_dir, trial.val_procedure[0])
    l = p.args[0][0].args[3]
    print(trial.get_from_record("labels"))
    print(type(maxSizeDict[labels]))
    for li in l:
        maxSizeDict[labels][li["name"]] = li["max_size"]
    print(maxSizeDict[labels])
    #print([(,l["max_size"]) for l in l])
    


from CMS_SURF_2016.utils.preprocessing import *
from CMS_SURF_2016.utils.batch import batchAssertArchived, batchExecuteAndTestTrials
def findsubsets(S):
    '''Finds all subsets of a set S'''
    out = []
    for m in range(2, len(S)):
        out = out + [set(x) for x in itertools.combinations(S, m)]
    return out
DELPHES_DIR = "/data/shared/Delphes/"
SOFTWAR_DIR = "/data/shared/Software/"
STORE_TYPE = "h5"
label_dir_pairs =             [   ("ttbar", DELPHES_DIR+"ttbar_lepFilter_13TeV/pandas_"+STORE_TYPE+"/"),
                ("wjet",  DELPHES_DIR+"wjets_lepFilter_13TeV/pandas_"+STORE_TYPE+"/"),
                ("qcd", DELPHES_DIR+"qcd_lepFilter_13TeV/pandas_"+STORE_TYPE+"/")
            ]
sort_on = "PT_ET"
max_EFlow_size = 100
ldpsubsets = [sorted(list(s)) for s in findsubsets(label_dir_pairs)]
#Make sure that we do 3-way classification as well
ldpsubsets.append(label_dir_pairs)
val_by_class = {}
for ldp in ldpsubsets:
    labels = [x[0] for x in ldp]
    observ_types = ['E/c', 'Px', 'Py', 'Pz', 'PT_ET','Eta', 'Phi', 'Charge', 'X', 'Y', 'Z',
                         'Dxy', 'Ehad', 'Eem', 'MuIso', 'EleIso', 'ChHadIso','NeuHadIso','GammaIso']
    object_profiles = [ObjectProfile("Electron",maxSizeDict[tuple(labels)]["Electron"]),
                        ObjectProfile("MuonTight", maxSizeDict[tuple(labels)]["MuonTight"]),
                        ObjectProfile("Photon", maxSizeDict[tuple(labels)]["Photon"]),
                        ObjectProfile("MissingET", maxSizeDict[tuple(labels)]["MissingET"]),
                        ObjectProfile("EFlowPhoton",max_EFlow_size, pre_sort_columns=["PT_ET"], pre_sort_ascending=False), 
                        ObjectProfile("EFlowNeutralHadron",max_EFlow_size, pre_sort_columns=["PT_ET"], pre_sort_ascending=False), 
                        ObjectProfile("EFlowTrack",max_EFlow_size, pre_sort_columns=["PT_ET"], pre_sort_ascending=False)] 
    #print(object_profiles)
    resolveProfileMaxes(object_profiles, ldp)

    dps, l = getGensDefaultFormat(archive_dir, (100000,20000,20000), 140000,                          object_profiles,ldp,observ_types,megabytes=500, verbose=0)
    #for dp in dps:
    #    dp.remove_from_archive()
    dependencies = batchAssertArchived(dps)
    train, num_train = l[0]
    val,   num_val   = l[1]
    test,  num_test  = l[2]
    max_q_size = l[3]
    val_by_class[tuple(labels)] = (val,num_val)
print(val_by_class)

data = np.zeros(( len(labelGroups[0]),len(labelGroups) ) ).tolist()
columns = [None] * len(labelGroups)
rows = [" "+ n + " " for n in names]

for j , tup in enumerate(labelGroups):
    for i, b in enumerate(tup):
        b.summary()
        labels = b.get_from_record("labels")
        columns[j] = str(tuple([str(x) for x in labels]))
        val_data,num_samples = val_by_class[tuple(labels)]
        error = getError(b,data=val_data,num_samples=num_samples,custom_objects={"Slice":Slice, "Lorentz" : Lorentz}, ignoreAssert=True)
        d = "%.5f %s %.5f" % (b.get_from_record("val_acc"),unichr(177), error)
        data[i][j] = d
        
        

print(repr(data))
plotTable(rows, columns, data,scale=2, title="Validation Accuracy for Lorentz Trials by Classification and Model")

def getTrialBins(trial, bins=20):
    trial.summary()
    labels = trial.get_from_record("labels")
    #print(labels)
    val, num = val_by_class[tuple(labels)]
    #print(val.args[0][0].args)
    d = accVsEventChar(trial, val, np.sum, "PT_ET", ["EFlowPhoton","EFlowNeutralHadron","EFlowTrack"],
                       bins=bins,num_samples=num,
                       custom_objects={"Slice": Slice, "Lorentz": Lorentz},equalBins=False)
    #plotBins(d,title='Accuracy vs Sum of PF Candidate PT', xlabel="PT GeV", ylabel='Accuracy', color=(0.553,0.188,0.38))
    return d
lorentz_w_bins = [getTrialBins(t) for t in top_lorentz_weighted]

colors = [(0,0,1.0),(.25,.75,.25), (1,0,0), (1,.65,0)]
lables = [trial.get_from_record("labels") for trial in top_lorentz_weighted]
plotBins(lorentz_w_bins,mode="scatter",title='Lorentz Weighted: Accuracy vs Sum of PF Candidate PT',binLabels=lables, xlabel="PT GeV", ylabel='Accuracy',
         legendTitle="Classification",colors=colors, alpha=.2,min_samples=0, ylim=(0.6, 1.025), xlim=(0,3000))

trial = lorentz[5]
history = trial.get_history()
trial.summary()
plot_history([("(qcd, wjet)",history)], plotLoss=True, title="Loss vs epoch for W boson + jets vs QCD")



