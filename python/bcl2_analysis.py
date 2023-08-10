import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import rstoolbox
import pandas as pd
import seaborn as sns
import numpy as np
import copy

sns.set(font_scale=1.5)

dlistT2T = rstoolbox.api.read_rosetta_silent("bcl2/3lhp_vs_4oyd", "t2t")
definition = {
    "scores":{
        "description": "description", "design_type": "type", "total_score": "score",
        "GlobalHRMSD": "GlobalHRMSD", "LocalHRMSD": "LocalHRMSD"
    }
}
dfT2T = rstoolbox.api.process_from_definitions(dlistT2T, definition)
dfT2T

def merge_rmsd_types( df, keys ):
    dataframes = []
    for k in keys["split"]:
        colIDs = copy.copy(keys["keep"])
        colIDs.append(k[0])
        wdf = df[colIDs]
        wdf = wdf.assign(temporarykey1=pd.Series([k[1]]*len(wdf[colIDs[0]])).values).copy(True)
        wdf = wdf.rename(index=str, columns={
            k[0]: keys["names"][0],
            "temporarykey1": keys["names"][1]
        })
        if ( len(k) > 2 ):
            wdf = wdf.assign(temporarykey2=pd.Series([k[2]]*len(wdf[colIDs[0]])).values).copy(True)
            wdf = wdf.rename(index=str, columns={
                "temporarykey2": keys["names"][2]
            })
        dataframes.append(wdf)
    return pd.concat(dataframes)

experiments = [
    ("nocst",   "nobinder", "bcl2/nocst/bcl2_nocst_nobinder_2_minisilent.gz"),
    ("nocst",   "binder",   "bcl2/nocst/bcl2_nocst_binder_2_minisilent.gz"),
    ("ssecst",  "nobinder", "bcl2/ssecst/bcl2_ssecst_nobinder_2_minisilent.gz"),
    ("ssecst",  "binder",   "bcl2/ssecst/bcl2_ssecst_binder_2_minisilent.gz"),
    ("fullcst", "nobinder", "bcl2/fullcst/bcl2_fullcst_nobinder_2_minisilent.gz"),
    ("fullcst", "binder",   "bcl2/fullcst/bcl2_fullcst_binder_2_minisilent.gz")
]
selector_nobinder = {
    "scores":{
        "description": "description", "design_type": "type", "score": "score", "GRMSD2Target": "GRMSD2Target",
        "GRMSD2Template": "GRMSD2Template", "LHRMSD2Target": "LHRMSD2Target", "LRMSD2Target": "LRMSD2Target"
    }
}
selector_binder = {
    "scores":{
        "description": "description", "design_type": "type", "design_score": "score", "GRMSD2Target": "GRMSD2Target",
        "GRMSD2Template": "GRMSD2Template", "LHRMSD2Target": "LHRMSD2Target", "LRMSD2Target": "LRMSD2Target"
    }
}
logic = {
    "keep": ["description", "type", "score", "condition"],
    "split": [("GRMSD2Target", "global", "target"), ("GRMSD2Template", "global", "template"),
              ("LHRMSD2Target", "hlocal", "target"), ("LRMSD2Target", "local", "target") ],
    "names": ["rmsd", "rmsd_type", "rmsd_to"]
}
dataframes     = []
plotdataframes = []
for experiment in experiments:
    dlist      = rstoolbox.api.read_rosetta_silent(experiment[2], experiment[0])
    definition = selector_nobinder if experiment[1] == "nobinder" else selector_binder
    df         = rstoolbox.api.process_from_definitions(dlist, definition)
    df = df.assign(condition=pd.Series([experiment[1]]*len(df["type"])).values)
    dataframes.append(df)
    plotdataframes.append(merge_rmsd_types( df, logic ))
    
data  = pd.concat(dataframes)
pdata = pd.concat(plotdataframes) 

g = sns.FacetGrid(pdata[(pdata["rmsd_type"] == "global")], col="type", size=9, aspect=0.5)
(g.map(sns.boxplot, "condition", "rmsd", "rmsd_to", showfliers=False)
 .despine(left=True).add_legend(title="RMSD Target"))
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Global RMSD')
sns.plt.show()

g = sns.FacetGrid(pdata[(pdata["rmsd_to"] == "target")], col="type", size=9, aspect=0.5)
(g.map(sns.boxplot, "condition", "rmsd", "rmsd_type", showfliers=False)
 .despine(left=True).add_legend(title="RMSD Type"))
plt.subplots_adjust(top=0.9)
g.fig.suptitle('RMSD with the Target')
sns.plt.show()

g = sns.FacetGrid(pdata[(pdata["rmsd_to"] == "target") & (pdata["condition"] == "binder")],
                  col="type", size=9, aspect=0.5)
(g.map(sns.boxplot, "condition", "rmsd", "rmsd_type", showfliers=False)
 .despine(left=True).add_legend(title="RMSD Type"))
plt.subplots_adjust(top=0.9)
g.fig.suptitle('RMSD with the Target')
sns.plt.show()

g = sns.FacetGrid(data, col="type", size=9, aspect=0.5)
g = g.map(sns.boxplot, "condition", "GRMSD2Target", showfliers=False)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Global RMSD with the Target')
sns.plt.show()

g = sns.FacetGrid(data, col="type", size=9, aspect=0.5)
g = g.map(sns.boxplot, "condition", "LRMSD2Target", showfliers=False)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Local RMSD with the Target')
sns.plt.show()

binderdata = data[(data["condition"] == "binder") & (data["LRMSD2Target"] < 20) & (data["GRMSD2Target"] < 10)]
g = sns.FacetGrid(binderdata, col="type", size=9, aspect=0.5)
g = g.map(sns.regplot, "GRMSD2Target", "LRMSD2Target", fit_reg=False)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Global vs. Local RMSD with the Target')
sns.plt.show()

# Let's setup a selection of decoys on the fullcst.binder dataset
fullcstbinderdata = data[(data["condition"] == "binder") & (data["type"] == "fullcst")]
max_display_limit = 6

grid = sns.JointGrid("GRMSD2Target", "LRMSD2Target", data=fullcstbinderdata,
                     xlim=(0,max_display_limit), ylim=(0,max_display_limit), size=9, ratio=50)
grid.plot_joint(plt.scatter)
plt.subplots_adjust(top=0.95)
sele_line_p1 = [0, 2.351]
sele_line_p2 = [1.605, 2.351]
selection_line = [
    [sele_line_p1[0], sele_line_p2[0]],
    [sele_line_p1[1], sele_line_p2[1]]
]
plt.plot(selection_line[0], selection_line[1], linewidth=2, color="red")
sele_line_p1 = [1.605, 2.351]
sele_line_p2 = [1.605, 0]
selection_line = [
    [sele_line_p1[0], sele_line_p2[0]],
    [sele_line_p1[1], sele_line_p2[1]]
]
plt.plot(selection_line[0], selection_line[1], linewidth=2, color="red")
grid.fig.suptitle('2WH6 as MOTIF SOURCE: classicFFL')
sns.plt.show()

top_rmsd_selection = 2.01
selected_data = data[(data["condition"] == "binder") & (data["type"] == "fullcst") &
                     (data["LRMSD2Target"] < top_rmsd_selection) & (data["GRMSD2Target"] < top_rmsd_selection) ]
selected_data

top_rmsd_selection_limit = 2
lineardata = [
    go.Parcoords(
        line = dict(color = 'blue'),
        dimensions = list([
            dict(range = [min(selected_data['score']), min(max(selected_data['score']),0)],
                label = 'Score', values = selected_data['score']),
            dict(range = [0,top_rmsd_selection_limit],
                label = 'RMSD2Template', values = selected_data['GRMSD2Template']),
            dict(range = [0,top_rmsd_selection_limit],
                label = 'RMSD2Target', values = selected_data['GRMSD2Target']),
            dict(range = [0,top_rmsd_selection_limit],
                label = 'Local RMSD2Target', values = selected_data['LRMSD2Target']),
            dict(range = [0,top_rmsd_selection_limit],
                label = 'Local RMSD2Target (helix)', values = selected_data['LHRMSD2Target'])
            
        ])
    )
]
plotly.offline.iplot(lineardata)

experiments = [
    ("ssecst",  "binder", "bcl2/ssecst/bcl2_ssecst_binder_from4oyd_3_minisilent.gz"),
    ("fullcst", "binder", "bcl2/fullcst/bcl2_fullcst_binder_from4oyd_3_minisilent.gz")
]
definition = {
    "scores":{
        "description": "description", "design_type": "type", "design_score": "score", "GRMSD2Target": "GRMSD2Target",
        "GRMSD2Template": "GRMSD2Template", "LHRMSD2Target": "LHRMSD2Target", "LRMSD2Target": "LRMSD2Target"
    }
}
logic = {
    "keep": ["description", "type", "score", "condition"],
    "split": [("GRMSD2Target", "global", "target"), ("GRMSD2Template", "global", "template"),
              ("LHRMSD2Target", "hlocal", "target"), ("LRMSD2Target", "local", "target") ],
    "names": ["rmsd", "rmsd_type", "rmsd_to"]
}
dataframes     = []
plotdataframes = []
for experiment in experiments:
    dlist      = rstoolbox.api.read_rosetta_silent(experiment[2], experiment[0])
    df         = rstoolbox.api.process_from_definitions(dlist, definition)
    df = df.assign(condition=pd.Series([experiment[1]]*len(df["type"])).values)
    dataframes.append(df)
    plotdataframes.append(merge_rmsd_types( df, logic ))
    
datar3  = pd.concat(dataframes)
pdatar3 = pd.concat(plotdataframes) 

g = sns.FacetGrid(pdatar3[(pdatar3["rmsd_to"] == "target")], col="type", size=9, aspect=0.5)
(g.map(sns.boxplot, "condition", "rmsd", "rmsd_type",)
 .despine(left=True).add_legend(title="RMSD Type"))
plt.subplots_adjust(top=0.9)
g.fig.suptitle('RMSD with the Target')
sns.plt.show()

binderdatar3 = datar3[(datar3["condition"] == "binder")]
g = sns.FacetGrid(binderdatar3, col="type", size=9)
g = g.map(sns.regplot, "GRMSD2Target", "LRMSD2Target", fit_reg=False)
plt.subplots_adjust(top=0.9)
g.axes[0,0].set_ylim(0,)
g.axes[0,0].set_xlim(0,)
g.axes[0,1].set_ylim(0,)
g.axes[0,1].set_xlim(0,)
g.fig.suptitle('Global vs. Local RMSD with the Target')
sns.plt.show()

binderdatar3 = datar3[(datar3["condition"] == "binder")
                      & (datar3["LRMSD2Target"] < 4) & (datar3["GRMSD2Target"] < 4)]
g = sns.FacetGrid(binderdatar3, col="type", size=9)
g = g.map(sns.regplot, "GRMSD2Target", "LRMSD2Target", fit_reg=False)
plt.subplots_adjust(top=0.9)
g.axes[0,0].set_ylim(0,)
g.axes[0,0].set_xlim(0,)
g.axes[0,1].set_ylim(0,)
g.axes[0,1].set_xlim(0,)
g.fig.suptitle('Global vs. Local RMSD with the Target (Zoom In)')
sns.plt.show()

experiments = [
    ("ssecst",  "binder", "bcl2/ssecst/bcl2_ssecst_binder_from4oyd_nodesign_4_minisilent.gz"),
    ("fullcst", "binder", "bcl2/fullcst/bcl2_fullcst_binder_from4oyd_nodesign_4_minisilent.gz")
]
definition = {
    "scores":{
        "description": "description", "design_type": "type", "design_score": "score", "GRMSD2Target": "GRMSD2Target",
        "GRMSD2Template": "GRMSD2Template", "LHRMSD2Target": "LHRMSD2Target", "LRMSD2Target": "LRMSD2Target"
    }
}
logic = {
    "keep": ["description", "type", "score", "condition"],
    "split": [("GRMSD2Target", "global", "target"), ("GRMSD2Template", "global", "template"),
              ("LHRMSD2Target", "hlocal", "target"), ("LRMSD2Target", "local", "target") ],
    "names": ["rmsd", "rmsd_type", "rmsd_to"]
}
dataframes     = []
plotdataframes = []
for experiment in experiments:
    dlist      = rstoolbox.api.read_rosetta_silent(experiment[2], experiment[0])
    df         = rstoolbox.api.process_from_definitions(dlist, definition)
    df = df.assign(condition=pd.Series([experiment[1]]*len(df["type"])).values)
    dataframes.append(df)
    plotdataframes.append(merge_rmsd_types( df, logic ))
    
dataND  = pd.concat(dataframes)
pdataND = pd.concat(plotdataframes) 

binderdataND = dataND[(dataND["condition"] == "binder")]
g = sns.FacetGrid(binderdataND, col="type", size=9)
g = g.map(sns.regplot, "GRMSD2Target", "LRMSD2Target", fit_reg=False)
plt.subplots_adjust(top=0.9)
g.axes[0,0].set_ylim(0,)
g.axes[0,0].set_xlim(0,)
g.axes[0,1].set_ylim(0,)
g.axes[0,1].set_xlim(0,)
g.fig.suptitle('Global vs. Local RMSD with the Target w/o Design')
sns.plt.show()

experiments = [
    ("fullcst", "minimized", "bcl2/fullcst/bcl2_fullcst_binder_from4oyd_minimize_5_minisilent.gz"),
    ("fullcst", "fulldesign", "bcl2/fullcst/bcl2_fullcst_binder_from4oyd_fullFastDesign_6_minisilent.gz")
]
definition = {
    "scores":{
        "description": "description", "design_type": "type", "design_score": "score", "GRMSD2Target": "GRMSD2Target",
        "GRMSD2Template": "GRMSD2Template", "LHRMSD2Target": "LHRMSD2Target", "LRMSD2Target": "LRMSD2Target"
    }
}
logic = {
    "keep": ["description", "type", "score", "condition"],
    "split": [("GRMSD2Target", "global", "target"), ("GRMSD2Template", "global", "template"),
              ("LHRMSD2Target", "hlocal", "target"), ("LRMSD2Target", "local", "target") ],
    "names": ["rmsd", "rmsd_type", "rmsd_to"]
}
dataframes     = []
plotdataframes = []
for experiment in experiments:
    dlist      = rstoolbox.api.read_rosetta_silent(experiment[2], experiment[0])
    df         = rstoolbox.api.process_from_definitions(dlist, definition)
    df = df.assign(condition=pd.Series([experiment[1]]*len(df["type"])).values)
    dataframes.append(df)
    plotdataframes.append(merge_rmsd_types( df, logic ))
    
dataMIN  = pd.concat(dataframes)
pdataMIN = pd.concat(plotdataframes) 

g = sns.FacetGrid(dataMIN, col="condition", size=9)
g = g.map(sns.regplot, "GRMSD2Target", "LRMSD2Target", fit_reg=False)
plt.subplots_adjust(top=0.9)
g.axes[0,0].set_ylim(0,)
g.axes[0,0].set_xlim(0,)
g.axes[0,1].set_ylim(0,)
g.axes[0,1].set_xlim(0,)
g.fig.suptitle('Global vs. Local RMSD with the Target w/ Global Minimize/Relax')
sns.plt.show()

g = sns.FacetGrid(dataMIN, col="condition", size=9)
g = g.map(sns.regplot, "GRMSD2Target", "LRMSD2Target", fit_reg=False)
plt.subplots_adjust(top=0.9)
g.axes[0,0].set_ylim(0,4)
g.axes[0,0].set_xlim(0,4)
g.fig.suptitle('Global vs. Local RMSD with the Target w/ Global Minimize/Relax (Zoom)')
sns.plt.show()

g = sns.lmplot(x="GRMSD2Target", y="LRMSD2Target", data=dataMIN[dataMIN["condition"] == "minimized"],
               fit_reg=False, size=10)
plt.subplots_adjust(top=0.95)
g.axes[0,0].set_ylim(0,4)
g.axes[0,0].set_xlim(0,4)
#g.axes[0,0].set_yticks(np.arange(0,5.5,0.5))
sele_line_p1 = [0, 2.351]
sele_line_p2 = [1.605, 2.351]
selection_line = [
    [sele_line_p1[0], sele_line_p2[0]],
    [sele_line_p1[1], sele_line_p2[1]]
]
plt.plot(selection_line[0], selection_line[1], linewidth=2, color="red")
sele_line_p1 = [1.605, 2.351]
sele_line_p2 = [1.605, 0]
selection_line = [
    [sele_line_p1[0], sele_line_p2[0]],
    [sele_line_p1[1], sele_line_p2[1]]
]
plt.plot(selection_line[0], selection_line[1], linewidth=2, color="red")
g.fig.suptitle('4YOD as MOTIF SOURCE: minimize')
sns.plt.show()

g = sns.lmplot(x="GRMSD2Target", y="LRMSD2Target", data=dataMIN[dataMIN["condition"] == "fulldesign"],
               fit_reg=False, size=10)
plt.subplots_adjust(top=0.95)
g.axes[0,0].set_ylim(0,4)
g.axes[0,0].set_xlim(0,4)
#g.axes[0,0].set_yticks(np.arange(0,5.5,0.5))
sele_line_p1 = [0, 2.351]
sele_line_p2 = [1.605, 2.351]
selection_line = [
    [sele_line_p1[0], sele_line_p2[0]],
    [sele_line_p1[1], sele_line_p2[1]]
]
plt.plot(selection_line[0], selection_line[1], linewidth=2, color="red")
sele_line_p1 = [1.605, 2.351]
sele_line_p2 = [1.605, 0]
selection_line = [
    [sele_line_p1[0], sele_line_p2[0]],
    [sele_line_p1[1], sele_line_p2[1]]
]
plt.plot(selection_line[0], selection_line[1], linewidth=2, color="red")
g.fig.suptitle('4YOD as MOTIF SOURCE: fullfastdesign')
sns.plt.show()

experiments = [
    ("fullcst", "minimized", "bcl2/fullcst/bcl2_fullcst_binder_from4oyd_minimize_5_minisilent.gz"),
    ("fullcst", "standard",  "bcl2/fullcst/bcl2_fullcst_binder_from4oyd_3_minisilent.gz")
]
definition = {
    "scores":{
        "description": "description", "design_type": "type", "design_score": "score", "time": "time"
    }
}
dataframes     = []
for experiment in experiments:
    dlist      = rstoolbox.api.read_rosetta_silent(experiment[2], experiment[0])
    df         = rstoolbox.api.process_from_definitions(dlist, definition)
    df = df.assign(condition=pd.Series([experiment[1]]*len(df["type"])).values)
    dataframes.append(df)
    
dataTIME  = pd.concat(dataframes)

binderdataTIME = dataTIME[(dataTIME["type"] == "fullcst")]
g = sns.FacetGrid(binderdataTIME, col="condition", size=9)
g = g.map(sns.regplot, "score", "time", fit_reg=False)
plt.subplots_adjust(top=0.9)
g.axes[0,0].set_ylim(0,)
#g.axes[0,0].set_xlim(,0)
g.fig.suptitle('Design Score vs. Time')
sns.plt.show()

experiments = [
    ("target", "minimized", "bcl2/target/bcl2_target_minimize_1_minisilent.gz"),
    ("target", "standard", "bcl2/target/bcl2_target_FflStandard_1_24233"),
    ("target", "full", "bcl2/target/bcl2_target_FflAllDesign_1_21529")
]
definition = {
    "scores":{
        "description": "description", "design_type": "type", "design_score": "score", "GRMSD2Target": "GRMSD2Target",
        "LHRMSD2Target": "LHRMSD2Target", "LRMSD2Target": "LRMSD2Target"
    }
}
dataframes     = []
for experiment in experiments:
    dlist      = rstoolbox.api.read_rosetta_silent(experiment[2], experiment[0])
    df         = rstoolbox.api.process_from_definitions(dlist, definition)
    df = df.assign(condition=pd.Series([experiment[1]]*len(df["type"])).values)
    dataframes.append(df)
    
dataTARGET = pd.concat(dataframes)

g = sns.FacetGrid(dataTARGET, col="condition", size=9)
g = g.map(sns.regplot, "GRMSD2Target", "LRMSD2Target", fit_reg=False)
plt.subplots_adjust(top=0.9)
g.axes[0,0].set_ylim(0,)
g.axes[0,0].set_xlim(0,)
g.axes[0,1].set_ylim(0,)
g.axes[0,1].set_xlim(0,)
g.axes[0,2].set_ylim(0,)
g.axes[0,2].set_xlim(0,)
g.fig.suptitle('Global vs. Local RMSD of the Target')
sns.plt.show()

experiments = [
    ("target", "FFLstandard", "bcl2/target/bcl2_target_FflStandard_1_24233"),
    ("design", "FFLstandard", "bcl2/fullcst/bcl2_fullcst_binder_from4oyd_3_minisilent.gz"),
    ("target", "FFLall", "bcl2/target/bcl2_target_FflAllDesign_1_21529"),
    ("design", "FFLall", "bcl2/fullcst/bcl2_fullcst_binder_from4oyd_fullFastDesign_6_minisilent.gz")
]
definition = {
    "scores":{
        "description": "description", "design_type": "type", "design_score": "score", "GRMSD2Target": "GRMSD2Target",
        "LHRMSD2Target": "LHRMSD2Target", "LRMSD2Target": "LRMSD2Target"
    }
}
dataframes     = []
for experiment in experiments:
    dlist      = rstoolbox.api.read_rosetta_silent(experiment[2], experiment[0])
    df         = rstoolbox.api.process_from_definitions(dlist, definition)
    df = df.assign(condition=pd.Series([experiment[1]]*len(df["type"])).values)
    dataframes.append(df)
    
dataCOMPARE = pd.concat(dataframes)

g = sns.lmplot(x="GRMSD2Target", y="LRMSD2Target", col="condition", hue="type", data=dataCOMPARE,
               sharey=True, sharex=True, fit_reg=False)
plt.subplots_adjust(top=0.85)
g.axes[0,0].set_ylim(0,)
g.axes[0,0].set_xlim(0,)
g.fig.suptitle('Target variability vs. Design Variability')
sns.plt.show()

g = sns.lmplot(x="GRMSD2Target", y="LRMSD2Target", col="condition", hue="type", data=dataCOMPARE,
               sharey=True, sharex=True, fit_reg=False)
plt.subplots_adjust(top=0.85)
g.axes[0,0].set_ylim(0,3)
g.axes[0,0].set_xlim(0,2)
g.fig.suptitle('Target variability vs. Design Variability (Zoom)')
sns.plt.show()

experiments = [
    ("target", "FFLstandard", "bcl2/target/bcl2_target_FflStandard_1_24233"),
    ("binder", "FFLstandard", "bcl2/fullcst/bcl2_fullcst_binder_from4oyd_3_minisilent.gz"),
    ("nobinder", "FFLstandard",  "bcl2/fullcst/bcl2_fullcst_nobinder_2_minisilent.gz")
]
selector_binder = {
    "scores":{
        "description": "description", "design_type": "type", "design_score": "score", "GRMSD2Target": "GRMSD2Target",
        "LHRMSD2Target": "LHRMSD2Target", "LRMSD2Target": "LRMSD2Target"
    }
}
selector_nobinder = {
    "scores":{
        "description": "description", "design_type": "type", "score": "score", "GRMSD2Target": "GRMSD2Target",
        "LHRMSD2Target": "LHRMSD2Target", "LRMSD2Target": "LRMSD2Target"
    }
}
dataframes     = []
for experiment in experiments:
    dlist      = rstoolbox.api.read_rosetta_silent(experiment[2], experiment[0])
    definition = selector_nobinder if experiment[0] == "nobinder" else selector_binder
    df         = rstoolbox.api.process_from_definitions(dlist, definition)
    df = df.assign(condition=pd.Series([experiment[1]]*len(df["type"])).values)
    dataframes.append(df)
    
dataTRIO = pd.concat(dataframes)

#sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
g = sns.lmplot(x="GRMSD2Target", y="LRMSD2Target",hue="type", data=dataTRIO.iloc[::-1], fit_reg=False)

g.axes[0,0].set_xlim(0,5)
sns.plt.show()

#g.savefig("/Users/bonet/Downloads/test.png")



