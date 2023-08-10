import pandas as pd
import geopandas as gpd

from bokeh.io import show, output_notebook
from bokeh.plotting import figure
import bokeh.models as bm
import bokeh.palettes

# set up bokeh for jupyter notebook
output_notebook()

df = pd.read_csv("membres_par_pays_dep.csv", sep=";")
df = df[df["Pays"] == "FR"]
df.rename(columns={df.columns[2]: "Departement"}, inplace=True)
df.index = df.Departement.apply(lambda x: "%02d" % x)
df.index.name = "CODE_DEPT"
df.head()

gdf = gpd.GeoDataFrame.from_file("DEPARTEMENT/DEPARTEMENT.shp")
gdf = gdf[["CODE_DEPT", "NOM_CHF", "NOM_DEPT", "geometry"]]
gdf.set_index("CODE_DEPT", inplace=True)
gdf.sort_index(inplace=True)
gdf.head()

gdf2 = gdf.join(df["Nombre de membres"])
gdf2.rename(columns={"Nombre de membres": "membres"}, inplace=True)
gdf2.fillna(value=0., inplace=True)
gdf2.head()

geo_src = bm.GeoJSONDataSource(geojson=gdf2.to_json())

# set up a log colormap
cmap = bm.LogColorMapper(
    palette=bokeh.palettes.BuGn9[::-1], # reverse the palette
    low=0, 
    high=gdf2.membres.max()
)

# define web tools
TOOLS = "pan,wheel_zoom,box_zoom,reset,hover,save"

# set up bokeh figure
p = figure(
    title="Membres de l'AFPY en 2016", 
    tools=TOOLS,
    toolbar_location="below",
    x_axis_location=None, 
    y_axis_location=None, 
    width=500, 
    height=500
)

# remove the grid
p.grid.grid_line_color = None

# core part !
#    * add a patch for each polygon in the geo data frame
#    * fill color from column 'membres' using the color map defined above
p.patches(
    'xs', 'ys', 
    fill_alpha=0.7, 
    fill_color={'field': 'membres', 'transform': cmap},
    line_color='black', 
    line_width=0.5, 
    source=geo_src
)

# set up mouse hover informations
hover = p.select_one(bm.HoverTool)
hover.point_policy = 'follow_mouse'
hover.tooltips = [
    ('Département:', '@NOM_DEPT'), 
    ("Membres:", "@membres"), 
    ("Contact:", "??"), 
    ("Afpyro:", "True/False")
]

# add a color bar
color_bar = bm.ColorBar(
    color_mapper=cmap,
    ticker=bm.LogTicker(),
    title_text_align="left",
    location=(0, 0),
    border_line_color=None,
    title="Membres"
)
p.add_layout(color_bar, 'right')

# show plot
show(p)

from bokeh.plotting import output_file
output_file("afpy_france.html")

