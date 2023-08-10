import pandas as pd
import numpy as np
import holoviews as hv
hv.extension('bokeh')
import pandas as pd
from bokeh.models import HoverTool



def well_nr_to_coords(nr, nx=12, ny=8):
    """Converts well_nr_to well coordinates. 
    To be used with pandas dataframes that have a wellnumber column like this:
    
    wellcoords = df.well.apply(well_nr_to_coords)
    df = pd.concat([df, wellcoords],axis=1)
    """
    
    cols = [nx] + [i+1 for i in range(nx-1)]
    rows = [chr(i+ord('A')) for i in range(ny)]
    
    colindex = nr  % nx   
    rowindex = int((nr-1) / nx)   
    return pd.Series(dict(row=rows[rowindex], col=cols[colindex], rownum=rowindex))

def create_custom_tooltip(info_fields, impath, im_col_name, height=200, width=200):
    """ Generates a custom HTML string for a tooltip
    info_fields: list of columnnames specifying the columns which should be displayed in the tooltip
    impath: path where the images to include in the tooltips are located (string)
    im_col_name: column name that contains the names of the images to include in the tooltip (string)
    height: integer specifying image height
    width: integer ...
    """
    info_html = ""
    for colnames in info_fields:
        info_html += f"""
            <div>
                <span style="font-size: 15px;">{colnames} @{colnames} </span>
            </div>"""

    custom_ttip = f"""
    <div>
        <div>
            <img
                src="{impath}/@{im_col_name}" height="{height}" width="{width}"
                style="float: left; margin: 0px 15px 15px 0px;"
                border="2"
            ></img>
        </div>
        {info_html}
    </div>
    """
    return custom_ttip

# create an image to display in the tooltip
import skimage.io
img = 255-255*np.eye(300, dtype = np.uint8)
skimage.io.imsave("im.gif", img)

df = pd.DataFrame()
df["well"] = np.arange(1,97)
df["value1"] = np.random.rand(df.shape[0])
df["value2"] = np.random.rand(df.shape[0])
df["giffile"] = "im.gif"
df["description"] = "some text"
df["Platename"] = "exampleplate"
wellcoords = df.well.apply(well_nr_to_coords)
df = pd.concat([df, wellcoords],axis=1)

info_fields = ["description"]
impath = "."
im_col_name = "giffile"

custom_ttip = create_custom_tooltip(info_fields, impath, im_col_name, 200, 200)

hover = HoverTool( tooltips= custom_ttip)

sorted = df.sort_values(['row','col'], ascending=[False,True])
width = 600
height = int(width * 8/12)

get_ipython().run_line_magic('opts', "HeatMap [tools=[hover] logz=False colorbar=True width=width height=height toolbar='above' invert_yaxis=True ](cmap='bwr')")

dataset = hv.Dataset(sorted, vdims=["value1"])
hv.HeatMap(dataset, vdims=["value1","giffile", "description"],kdims=["col","row"]).redim.range(value1=(0.0,1.0))



