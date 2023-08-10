import shapely as shp
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
import geopandas as gpd
import descartes
import seaborn as sns
get_ipython().magic('matplotlib inline')

fname = 'Data/tracks_listing_households.geojson'
tracks = gpd.GeoDataFrame.from_file(fname)
print tracks.crs

totalshape = tracks.geometry[1]
for shape in tracks.geometry:
    totalshape = totalshape.union(shape)

totalshape

un = totalshape.envelope.symmetric_difference(totalshape)
un

gdf = gpd.GeoDataFrame.from_file('Data/jp-sample.geojson')
gdf.head()

df4

df0 = gdf[gdf['kmeans'] ==0 ]
df1 = gdf[gdf['kmeans'] ==1 ]
df2 = gdf[gdf['kmeans'] ==2 ]
df3 = gdf[gdf['kmeans'] ==3 ]
df4 = gdf

listofdf = [df0,df1,df2,df3,df4]
thenames = ['Normal People','The 2%', 'Central Action', 'Hip Kids','All Listings']

df = gdf[['latitude','longitude']]
coords = gdf.as_matrix(columns=['latitude','longitude'])

f, ax = plt.subplots(figsize=(55,55))
for polygon in un:

    patch = descartes.PolygonPatch(polygon, alpha=0.5, zorder=2)
    ax.add_patch(patch)
sns.jointplot(x="longitude", y="latitude", data=gdf, kind="kde", size=24,ax=ax)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap    
mycolor1 = ListedColormap('k')
mycolor2 = ListedColormap('w')


i=0
for df in listofdf:
    
    y = df['latitude'].as_matrix()
    x = df['longitude'].as_matrix()
    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    plt.rcParams['axes.facecolor'] = 'white'
    f, ax = plt.subplots(figsize=(55,55))
    plt.rcParams['axes.facecolor'] = 'white'
    for polygon in un:
        patch = descartes.PolygonPatch(polygon, alpha=0.5, zorder=2)
        ax.add_patch(patch)

    tracks[tracks['HD01_VD01']==0].plot(c = 'k', linewidth = .5, ax = ax, cmap=mycolor1, alpha=1)
    tracks[tracks['HD01_VD01']>0].plot(c = 'k', linewidth = .5, ax = ax,  cmap = mycolor2, alpha=.15)
    ax.scatter(x, y, c=z, s=50, edgecolor='', cmap=plt.cm.jet )
    plt.axis('off')
    plt.title("NYC Density of Airbnb listings Cluster: "+str(i)+', '+thenames[i], size=70)
    i+=1
    normalize = mcolors.Normalize(vmin=min(z),vmax=max(z))
    colormap = cm.jet
    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(df.assign(cl = z))
    cbar = plt.colorbar(scalarmappaple, ax=ax)
    cbar.ax.tick_params(labelsize=60)
    plt.rcParams['axes.facecolor'] = 'white'
    plt.show()
    plt.rcParams['axes.facecolor'] = 'white'



features =[u'Median_income',
     #u'Median_rent',
     #u'accesstosubway',
     #u'availability_365',
     u'beer_count',
     #u'boro_ct_2010',
     #u'calculated_host_listings_count',
     u'coffee_count',
     u'connectivityScore',
     #'geometry',
     #u'host_id',
     #u'host_name',
     #u'id',
     #u'index_left',
     #u'last_review',
     #u'latitude',
     #u'longitude',
     #u'minimum_nights',
     #u'name',
     #u'neighbourhood',
     #u'neighbourhood_group',
     #u'number_of_reviews',
     u'price',
     #u'reviews_per_month',
     #u'room_type',
     #'Private_room',
     #'Entire_home/apt',
     #'Shared_room'
          ]
features

def violin_feature(feature, classifier):
    f, ax = plt.subplots(figsize=(5,5))
    sns.violinplot(x=classifier, y=feature, data=gdf, inner=None, ax=ax)
    ax.set_title('{0} distribution with {1}'.format(feature, classifier), fontsize=15)
    ax.set_xlabel('Categories', fontsize=15)
    plt.xticks( fontsize = 15)
    plt.yticks( fontsize = 15)
    ax.set_ylabel(feature, fontsize=15)

for feat in features:
    violin_feature(feat, 'kmeans')

def feature_dist(feature, classifier):
    for cid in set(gdf[classifier]):
        sns.distplot(gdf[gdf[classifier] == cid][feature], hist=False, label=str(cid))
        

    plt.legend()
    plt.show()

for feat in features:
    feature_dist(feat, 'kmeans')


ax = sns.distplot(gdf.connectivityScore)
ax.set(xlabel='Connectivity Score Distribution')

type(listofdf)

listofdf[0]

z

thenames


#import plotly.plotly as py
from plotly.graph_objs import *
from plotly.graph_objs import *

def scatt(x,y,z, name, text_str):
    data = Data([
    Scattermapbox(
        lat=y,
        lon=x,
        showlegend=True,
        name = thenames[count],
        mode='markers',
        marker=Marker(
            size=9,
            color = z,
            colorscale='Jet',
            colorbar = dict(
                thickness = 10,
                titleside = "right",
                #outlinecolor = "rgba(68, 68, 68, 0)",
                ticks = "outside",
                #ticklen = 3,
                #showticksuffix = "last",
                #ticksuffix = " inches",
                #dtick = 0.1
            )
        ),
        text=text_str,
    )])
    return data
    
mapbox_access_token = 'pk.eyJ1IjoibG1mNDQ1IiwiYSI6ImNqYzk2dGZoMTJ5Y3EzM3I1Zm9ha2pmbW8ifQ.yxcJVd1-XTC5lGXMlyHBiQ'
count = 0
fin_data = []
for df in listofdf[:]:
    
    y = df['latitude'].as_matrix()
    x = df['longitude'].as_matrix()
    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    name = thenames[count]
    text_str = df['neighbourhood'].as_matrix()[idx]
    fin_data.append(scatt(x,y,z, name,text_str))
    count+=1
    

layout = Layout(
    #autosize=True,
    autosize=False,
    width=900,
    height=900,
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=40.7668,
            lon=-73.9765 #40.766818, -73.976545
        ),
        pitch=0,
        zoom=10,
        style='dark'
    ),
    xaxis=dict(
        #showgrid=True,
        #zeroline=True,
        showline=True,
        mirror='ticks',
        #gridcolor='#bdbdbd',
        #gridwidth=2,
        #zerolinecolor='#969696',
        #zerolinewidth=4,
        linecolor='#636363',
        linewidth=6
    ),
    yaxis=dict(
        #showgrid=True,
        #zeroline=True,
        showline=True,
        mirror='ticks',
        #gridcolor='#bdbdbd',
        #gridwidth=2,
        #zerolinecolor='#969696',
        #zerolinewidth=4,
        linecolor='#636363',
        linewidth=6
    ),
)

fig = dict(data=fin_data[0], layout=layout)
py.iplot(fig, filename='Multiple test')

thenames

#import plotly.plotly as py
from plotly.graph_objs import *

def scatt(x,y,z, name, text_str):
    if name == 'All Listings':
        vis=True
    else:
        vis=False
    data = Scattermapbox(
        lat=y,
        lon=x,
        visible=vis,
        showlegend=True,
        name = thenames[count],
        mode='markers',
        marker=Marker(
            size=9,
            color = z,
            colorscale='Jet',
            colorbar = dict(
                thickness = 10,
                titleside = "right",
                #outlinecolor = "rgba(68, 68, 68, 0)",
                ticks = "outside",
                
                #ticklen = 3,
                #showticksuffix = "last",
                #ticksuffix = " inches",
                #dtick = 0.1
            )
        ),
        text=text_str,
    )
    return data
    
mapbox_access_token = 'pk.eyJ1IjoibG1mNDQ1IiwiYSI6ImNqYzk2dGZoMTJ5Y3EzM3I1Zm9ha2pmbW8ifQ.yxcJVd1-XTC5lGXMlyHBiQ'
count = 0
fin_data = []
for df in listofdf[:]:
    
    y = df['latitude'].as_matrix()
    x = df['longitude'].as_matrix()
    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    name = thenames[count]
    text_str = df['neighbourhood'].as_matrix()[idx]
    fin_data.append(scatt(x,y,z, name,text_str))
    count+=1
    
    
    
updatemenus = list([
    dict(type="buttons",
         active=-1,
         buttons=list([
            dict(label = thenames[4],
                 method = 'update',
                 args = [{'visible': [False, False, False, False, True]},
                         {'title': thenames[4],
                          #'annotations': high_annotations
                         }]),
            dict(label = thenames[0],
                 method = 'update',
                 args = [{'visible': [True, False, False, False,False]},
                         {'title': thenames[0],
                          #'annotations': high_annotations
                         }]),
            dict(label = thenames[1],
                 method = 'update',
                 args = [{'visible': [False, True, False, False, False]},
                         {'title': thenames[1],
                          #'annotations': high_annotations
                         }]),
            dict(label = thenames[2],
                 method = 'update',
                 args = [{'visible': [False, False, True, False, False]},
                         {'title': thenames[2],
                          #'annotations': high_annotations
                         }]),
            dict(label = thenames[3],
                 method = 'update',
                 args = [{'visible': [False, False, False, True, False]},
                         {'title': thenames[3],
                          #'annotations': high_annotations
                         }]),
        ]),
    )
])    

layout = Layout(
    #autosize=True,
    updatemenus=updatemenus,
    title='Airbnb Automated Classification System',
    autosize=False,
    width=900,
    height=900,
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=40.7668,
            lon=-73.9765 #40.766818, -73.976545
        ),
        pitch=0,
        zoom=10,
        style='light'
    ),
    xaxis=dict(
        #showgrid=True,
        #zeroline=True,
        showline=True,
        mirror='ticks',
        #gridcolor='#bdbdbd',
        #gridwidth=2,
        #zerolinecolor='#969696',
        #zerolinewidth=4,
        linecolor='#636363',
        linewidth=6
    ),
    yaxis=dict(
        #showgrid=True,
        #zeroline=True,
        showline=True,
        mirror='ticks',
        #gridcolor='#bdbdbd',
        #gridwidth=2,
        #zerolinecolor='#969696',
        #zerolinewidth=4,
        linecolor='#636363',
        linewidth=6
    ),
    legend=dict(orientation="h")
)

fig = dict(data=fin_data, layout=layout)
py.iplot(fig, filename='Airbnb_Automated_Classification_System')



