import pandas as pd
import numpy as np
import holoviews as hv
from bokeh.models import HoverTool
hv.extension('bokeh')
import pandas as pd
import re
from plateviewer_tools import well_nr_to_coords, create_custom_tooltip

def createDF():
    df = pd.DataFrame(np.random.randn(96, 4), columns=list('ABCD'))
    df["well"] = np.arange(96)+1
    # change the scaling
    df["A"] *= 255
    df["B"] *= 10e3
    df["C"] *= 30e6
    return df

df1 = createDF()
df1 = pd.melt(df1, id_vars="well", var_name="Platename", value_name="value1")
df2 = createDF()
df2 = pd.melt(df2, id_vars="well", var_name="Platename", value_name="value2")

df = pd.concat([df1, df2.value2], axis=1)
# create some outliers in a few plates
df.loc[df.well==20, "value2"] *= 30
# you may also want to introduce a few NaNs for testing
#df.loc[df.well==23, "value2"] = np.nan

wellcoords = df.well.apply(well_nr_to_coords)
df = pd.concat([df, wellcoords],axis=1)

gifs = pd.read_csv("giphy_urls.csv")
df = pd.concat([df, gifs[0:len(df)].giffile], axis=1)
# note ! not all URLs are unique. Expect to see the same gif in several wells !

df  = df.sort_values(['row','col'], ascending=[False,True])

df.head()

value_fields = ["value1", "value2", "well"]
info_fields = [ "well", "value1", "value2", "Platename"]
impath = ""
im_col_name = "giffile"
custom_tip = create_custom_tooltip(info_fields, impath, im_col_name, 100, 100)
hover = HoverTool( tooltips= custom_tip)

print(custom_tip)

width = 700
height = int(width * 8/12)

# from https://stackoverflow.com/questions/46024901/how-to-format-colorbar-label-in-bokeh
from bokeh.models import PrintfTickFormatter
formatter = PrintfTickFormatter(format='%1f')
colorbar_opts={'formatter': formatter}

def select_plate(Platename, Column, cmap, lower_percentile, upper_percentile):
    #print(f"Platename: {Platename}, Column: {Column}, Normalisation: {Normalisation}")
    tmp = df[df["Platename"]==Platename]
    values = tmp[Column]
    vdims = [Column] + list(set(info_fields + ["giffile"]) - set([Column]))
    kdims = ['col','row']
    cmap_range = (np.percentile(values, lower_percentile ), np.percentile(values, upper_percentile))             
    dataset = hv.Dataset(tmp, vdims=vdims, kdims=kdims)
    heatmap = hv.HeatMap(dataset, vdims=vdims, kdims=kdims).redim.range(**{Column: cmap_range}).options(tools=[hover], logz=False ,colorbar=True ,width=width, height=height, toolbar='below', colorbar_opts=colorbar_opts, invert_yaxis=True, cmap=cmap)
    return  heatmap
     
hv.DynamicMap(select_plate, kdims=['Platename','Column', 'cmap', 'lower_percentile','upper_percentile', ]).redim.values(Platename=pd.unique(df["Platename"]), Column=["value1", "value2"], cmap=["bwr", "coolwarm"], lower_percentile=list(range(50)),upper_percentile=list(range(50,101)))



