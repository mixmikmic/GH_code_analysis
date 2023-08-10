get_ipython().magic('matplotlib inline')
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from fuzzywuzzy import process

regions = gpd.GeoDataFrame.from_file('Original/Regions.shp')
provinces = gpd.GeoDataFrame.from_file('Original/PHL_adm1.shp')
psgg_code = pd.read_csv('psgg_codes.csv', dtype=object)

regions.shape

regions

provinces.shape

provinces.columns

cols = ['NAME_1', 'geometry']
provinces.head()[cols]

neg_prov = provinces.loc[
    provinces.NAME_1.str.contains(r'Negros'), cols
]
neg_prov

psgg_code.shape

psgg_code

def get_psgg_code(orig_name):
    if orig_name.startswith('Metropolitan'):
        return '13'
    else:
        choice, _ = process.extractOne(orig_name, psgg_code.region.values)
        p_code = psgg_code.psgg_code[psgg_code.region == choice]
        return p_code.values[0]

regions['psgg_code'] = regions['REGION'].apply(lambda x: get_psgg_code(x))

regions

regions.set_index('psgg_code', inplace=True)
regions

# create a series with the region numbers/abbreviations to indicate map locations
def make_map_text(name):
    text = name.split(' ')
    return text[1] if name.startswith('Region') else text[0]

map_names = psgg_code.loc[:, ['psgg_code', 'region']]
map_names['region'] = map_names['region'].apply(lambda x: make_map_text(x))
map_names.set_index('psgg_code', inplace=True, drop=True)

# Plotting it
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 10))

regions.plot(ax=ax, color='forestgreen', linewidth=0)
regions[regions.index == '06'].plot(ax=ax, color='orange', linewidth=0)
regions[regions.index == '07'].plot(ax=ax, color='slateblue', linewidth=0)

for i, point in regions.centroid.iteritems():
    reg_n = map_names.loc[i, 'region']
    ax.text(s=reg_n, x=point.x, y=point.y, fontsize='large')

ax.set_title('PH Administrative Regions (pre-2015)')
ax.set_axis_off()

plt.savefig('map-01-previous.png', bbox_inches='tight')

# get the shapes of the two provinces (see input cell 8 and its output)
neg_oc = neg_prov.iloc[0, 1]                # Negros Occidental
neg_or = neg_prov.iloc[1, 1]                # Negros Oriental

# get shapes of regions vi and vii
reg_6 = regions.loc['06', 'geometry']
reg_7 = regions.loc['07', 'geometry']

# remove the provinces from their respective pre-2015 regions
regions.loc['06', 'geometry'] = reg_6.difference(neg_oc)
regions.loc['07', 'geometry'] = reg_7.difference(neg_or)

# combining the two provinces' shapes
neg_geom = neg_oc.union(neg_or)

# appending the new region to the regions GDF
neg_reg = gpd.GeoDataFrame([{'REGION': 'Negros Island Region (NIR)',
                             'geometry': neg_geom,
                             'psgg_code': '18'}],
                            columns=['REGION', 'geometry', 'psgg_code'])
neg_reg.set_index('psgg_code', inplace=True, drop=True)
regions = regions.append(neg_reg)
regions

regions.shape

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 10))

regions.plot(ax=ax, cmap='Dark2', linewidth=0)

for i, point in regions.centroid.iteritems():
    reg_n = map_names.loc[i, 'region']
    ax.text(s=reg_n, x=point.x, y=point.y, fontsize='large')

ax.set_title('PH Administrative Regions (Present)')
ax.set_axis_off()

plt.savefig('map-02-present.png', bbox_inches='tight')

regions.loc['13', 'REGION'] = 'National Capital Region (NCR)'
regions

regions.to_file('Updated/ph-regions-2015.shp', driver='ESRI Shapefile')

