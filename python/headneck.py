tablesPath = '/home/jovyan/data/QIN-HEADNECK-Tables'
#  set this to your location of the tables if running locally
#tablesPath = '/Users/fedorov/github/dcm2tables/Tables'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from bokeh.models import ColumnDataSource, OpenURL, TapTool
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook
from bokeh.colors import RGB

output_notebook()

SR1500_MeasurementGroups = pd.read_csv(tablesPath+'/SR1500_MeasurementGroups.tsv', sep='\t', low_memory=False)
SR1500_MeasurementGroups.columns

SR1500_Measurements = pd.read_csv(tablesPath+'/SR1500_Measurements.tsv', sep='\t', low_memory=False)
SR1500_Measurements.columns

SR1500_Measurements.shape

Measurements_merged = pd.merge(SR1500_Measurements,SR1500_MeasurementGroups,on=["SOPInstanceUID","TrackingUniqueIdentifier"])
Measurements_merged.shape

Measurements_merged.columns

(Measurements_merged["quantity_CodeMeaning"].map(str)+"_"+Measurements_merged["derivationModifier_CodeMeaning"].map(str)).unique()

Measurements_merged["Finding_CodeMeaning"].unique()

CompositeContext=pd.read_csv(tablesPath+'/CompositeContext.tsv', sep='\t',low_memory=False)
CompositeContext.columns

Measurements_merged.shape

Measurements_merged = pd.merge(Measurements_merged, CompositeContext, on="SOPInstanceUID")
Measurements_merged.shape

Measurements_merged.columns

from bokeh.models import ColumnDataSource, OpenURL, TapTool
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook
from bokeh.colors import RGB

from bokeh.models import HoverTool, PanTool, WheelZoomTool, BoxZoomTool, ResetTool, TapTool

output_notebook()

volume = []
user = []
method = []
sesssion = []
subject = []

#SR_merged = pd.merge(SR_merged, segReferences)


#subset = SR_merged[SR_merged["PersonObserverName"]=="User1"]
subset = Measurements_merged[Measurements_merged["Finding_CodeMeaning"]=="Neoplasm, Primary"]
subset = subset[subset["quantity_CodeMeaning"]=="Volume"]

print("Identifiers of the users: "+str(subset["PersonObserverName"].unique()))
print("Identifiers of the activity sessions: "+str(subset["activitySession"].unique()))

#subset = subset[subset["activitySession"]==1]
#subset = subset[subset["segmentationToolType"]=="SemiAuto"]

#subset.sort_values("value", inplace=True)

#subset=subset[subset["PatientID"]=="QIN-HEADNECK-01-0003"]

volumes = subset["value"].values
observers = subset["PersonObserverName"].values
subjects = subset["PatientID"].values

#subset["segmentationToolType"].unique()

colormap = {'User1': 'red', 'User2': 'green', 'User3': 'blue'}
colors = [colormap[x] for x in subset['PersonObserverName'].tolist()]

source = ColumnDataSource(data=dict(
    x=volumes,
    y=subjects,
    color=colors,
    labels = subset["PersonObserverName"].tolist()
    ))

hover = HoverTool(tooltips=[
    ("(Volume, Subject)", "($x, $y)")
])

wZoom = WheelZoomTool()
bZoom = BoxZoomTool()
reset = ResetTool()
pan = PanTool()

p = figure(x_range=[np.min(volumes),np.max(volumes)], y_range=subjects.tolist(),            tools = [hover, wZoom, bZoom, reset, pan],            title="Variability of primary neoplasm volume by reader")
p.yaxis.axis_label = "PatientID"
p.xaxis.axis_label = subset["quantity_CodeMeaning"].values[0]+', '+subset['units_CodeMeaning'].values[0]

p.circle('x','y',color='color',source=source, legend='labels')

p.legend.location = "bottom_right"

show(p)

References=pd.read_csv(tablesPath+'/References.tsv', sep='\t', low_memory=False)
References.columns

# 1.2.840.10008.5.1.4.1.1.66.4 is the SOPClassUID corresponding to the DICOM Segmentation image object
segReferences = References[References["ReferencedSOPClassUID"]=='1.2.840.10008.5.1.4.1.1.66.4']
segReferences = segReferences[["SOPInstanceUID","SeriesInstanceUID"]].rename(columns={"SeriesInstanceUID":"ReferencedSeriesInstanceUID"})

# I am not a pandas expert, so just to be safe, I check that the dimensions of the data frame 
# do not change after the merge operation ...
Measurements_merged.shape

Measurements_merged = pd.merge(Measurements_merged, segReferences)
Measurements_merged.shape

subset = Measurements_merged[Measurements_merged["Finding_CodeMeaning"]=="Neoplasm, Primary"]
subset = subset[subset["quantity_CodeMeaning"]=="Volume"]

volumes = subset["value"].values
observers = subset["PersonObserverName"].values
subjects = subset["PatientID"].values

colormap = {'User1': 'red', 'User2': 'green', 'User3': 'blue'}
colors = [colormap[x] for x in subset['PersonObserverName'].tolist()]

source = ColumnDataSource(data=dict(
    x=volumes,
    y=subjects,
    color=colors,
    labels = subset["PersonObserverName"].tolist(),
    seriesUID=subset["ReferencedSeriesInstanceUID"]
    ))

hover = HoverTool(tooltips=[
    ("(Volume, Subject)", "($x, $y)")
])

wZoom = WheelZoomTool()
bZoom = BoxZoomTool()
reset = ResetTool()
tap = TapTool()
pan = PanTool()

p = figure(x_range=[np.min(volumes),np.max(volumes)],            y_range=subjects.tolist(),            tools = [hover, wZoom, bZoom, reset, tap, pan],
           title="Variability of primary neoplasm volume by reader")

p.circle('x','y',color='color',source=source, legend='labels')

url = "http://pieper.github.com/dcmjs/examples/qiicr/?seriesUID=@seriesUID"
taptool = p.select(type=TapTool)
taptool.callback = OpenURL(url=url)

p.xaxis.axis_label = subset["quantity_CodeMeaning"].values[0]+', '+subset['units_CodeMeaning'].values[0]
p.legend.location = "bottom_right"

show(p)



