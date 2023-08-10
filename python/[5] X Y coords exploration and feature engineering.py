import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from utils_all import *
get_ipython().magic('matplotlib inline')
from matplotlib.pyplot import hist2d, xlim, ylim
import matplotlib.ticker as ticker
import seaborn as sns

get_ipython().magic('store -r DATA_NUM_CL_WITH_NAN_TXT')

data_cl = DATA_NUM_CL_WITH_NAN_TXT

data_xy = data_cl[['meta_name','x_coords','y_coords']]

data_xy.head()

def draw_hex(meta_name, df, ylim=(960,0), xlim=(0,1000)):
    dataXY = df[df.meta_name == meta_name]
    try:
        smpl = dataXY.sample(1000)
    except Exception as e:
        smpl = dataXY
    g = sns.jointplot(x=smpl.x_coords, y=smpl.y_coords, kind='scatter', ylim=ylim, xlim=xlim)

    g.ax_joint.xaxis.set_major_locator(ticker.MultipleLocator(100))
    g.ax_joint.yaxis.set_major_locator(ticker.MultipleLocator(100))

    plt.show()

draw_hex('location', data_xy)

draw_hex('startDate', data_xy)

draw_hex('name', data_xy)

draw_hex('description', data_xy)

data_rect = data_cl[['meta_name','x_coords','y_coords', 'block_height', 'block_width']]

data_rect['x1'] = data_rect.x_coords
data_rect['y1'] = data_rect.y_coords

data_rect['x2'] = data_rect.x_coords + data_rect.block_width
data_rect['y2'] = data_rect['y1']

data_rect['x3'] = data_rect['x1']
data_rect['y3'] = data_rect['y1'] + data_rect.block_height

data_rect['x4'] = data_rect['x3'] + data_rect.block_width
data_rect['y4'] = data_rect['y3']

data_block_centers = pd.DataFrame()
data_block_centers['x_coords'] = data_rect.x1 + (data_rect.x2 - data_rect.x1)/2
data_block_centers['y_coords'] = data_rect.y1 + (data_rect.y3 - data_rect.y1)/2
data_block_centers['meta_name'] = data_rect.meta_name

draw_hex('name', data_block_centers, ylim=(5000,0))

draw_hex('description', data_block_centers, ylim=(5000,0))

draw_hex('location', data_block_centers,  ylim=(5000,0))

draw_hex('startDate', data_block_centers,  ylim=(5000,0))

data_block_centers.columns = ['x_center', 'y_center', 'meta_name']

xy_data = pd.concat(axis=1, objs=
          [
            data_cl[['x_coords','y_coords', 'block_height', 'block_width']],
            data_block_centers
          ])

xy_data.groupby('meta_name').hist()

xy_data.groupby('meta_name').block_width.plot(kind='hist', alpha=0.5)

xy_data.groupby('meta_name').block_height.plot(kind='hist', alpha=0.5)

xy_data.groupby('meta_name').x_center.plot(kind='hist', alpha=0.5)

xy_data.groupby('meta_name').y_center.plot(kind='hist', alpha=0.4)

xy_data.groupby('meta_name').x_coords.plot(kind='hist', alpha=0.5)

xy_data.groupby('meta_name').y_coords.plot(kind='hist', alpha=0.5)

xy_data.groupby('meta_name').describe()

xy_data.shape

xy_centers = xy_data[['x_center','y_center']]

from sklearn.cluster import KMeans

xy_meta_ceanters = xy_data[['x_center','y_center', 'meta_name']]

xy_meta_ceanters = xy_meta_ceanters.fillna(xy_meta_ceanters.mean())

xy_name = xy_meta_ceanters[xy_meta_ceanters.meta_name == 'name'].drop_duplicates(subset=['x_center'])
xy_descr = xy_meta_ceanters[xy_meta_ceanters.meta_name == 'description'].drop_duplicates(subset=['x_center'])
xy_date = xy_meta_ceanters[xy_meta_ceanters.meta_name == 'startDate'].drop_duplicates(subset=['x_center'])
xy_not_event = xy_meta_ceanters[xy_meta_ceanters.meta_name == 'not_event_element'].drop_duplicates(subset=['x_center'])
xy_loc = xy_meta_ceanters[xy_meta_ceanters.meta_name == 'location'].drop_duplicates(subset=['x_center'])

estimators = {'xy_name_3': KMeans(n_clusters=3),
              'xy_name_8': KMeans(n_clusters=8)}

est = KMeans(n_clusters=3)

def cluster_viz(df, k, name):
    est = KMeans(n_clusters=k)
    df['cluster'] = est.fit_predict(df[['x_center','y_center']].values)
    df.cluster.value_counts()

    sns.lmplot('x_center', 'y_center',
               data=df,
               fit_reg=False,
               hue="cluster",  
               scatter_kws={"marker": "D", "s": 100})
    plt.title('X,Y coordinates of {} colored by corresponding cluster'.format(name))
    plt.gca().invert_yaxis()
    
    return df



xy_name = cluster_viz(xy_name, 3, 'name')

xy_descr = cluster_viz(xy_descr, 8, 'description')

xy_loc = cluster_viz(xy_loc, 8, 'location')

xy_not_event = cluster_viz(xy_not_event, 8, 'location')

data_cl = pd.concat([data_cl, xy_centers], axis=1)

DATA_NUM_CL_WITH_NAN_TXT_XY = data_cl

get_ipython().magic('store DATA_NUM_CL_WITH_NAN_TXT_XY')

import h5py
DATA_NUM_CL_WITH_NAN_TXT_XY.to_hdf('store.h5', 'DATA_NUM_CL_WITH_NAN_TXT_XY')



