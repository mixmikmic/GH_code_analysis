import brightway2 as bw
import bw2regional as reg
import os

import geopandas as gpd
import pandas as pd
import folium
import numpy as np

bw.projects.set_current('bw2_seminar_2017')

imp = bw.ExcelImporter("data/ethanol-inventory.xlsx")
imp.apply_strategies()
imp.match_database("ecoinvent 2.2", fields=('name', 'unit', 'location'))
imp.statistics()

imp.write_database()

bw.databases['ecoinvent 2.2']['geocollections'] = ['world', 'ecoinvent 2.2']
bw.databases['Sugarcane']['geocollections'] = ['world', 'ecoinvent 2.2']
bw.databases['biosphere3']['geocollections'] = []
bw.databases.flush()

reg.bw2regionalsetup()

from bw2_lcimpact import import_regionalized_lcimpact

import_regionalized_lcimpact()

reg.geocollections['sugarcane_landuse_intensity'] = {
    'filepath': os.path.abspath("data/sugarcane_landuse_intensity.tif"),
    'band': 1
}
reg.geocollections['sugarcane_water_intensity'] = {
    'filepath': os.path.abspath("data/sugarcane_water_intensity.tif"),
    'band': 1
}
reg.geocollections['weighted-pop-density'] = {
    'band': 1,
    'kind': 'raster',
    'sha256': '11ec180aaa8d1f68629c06a9c2e6eb185f8e1e4c0d6713bab7f9219f1d160644'
}

inters = [
    'world-topo-watersheds-hh',
    'world-topo-watersheds-eq-sw-core',
    'world-topo-watersheds-eq-sw-extended',
    'world-topo-particulate-matter',
    'world-topo-ecoregions',
]

crop_rasters = [
    'sugarcane_landuse_intensity',
    'sugarcane_water_intensity',
    'weighted-pop-density',
]

for x in inters:
    for y in crop_rasters:
        remote.rasterstats_as_xt(x, y, x + "-" + y)

crops = [x for x in bw.Database("ecoinvent 2.2") if 'sugarcane' in x['name']]

# Do agricultural activities with the sugarcane intensity map, 
# all others with the weighted pop density map
xt_ag = reg.ExtensionTablesLCA(
    {('Sugarcane', 'driving'): 1},
    ('LC-IMPACT', 'Land Use', 'Occupation', 'Marginal', 'Core'),
    xtable='world-topo-ecoregions-sugarcane_landuse_intensity',
    limitations={
        'activities': crops,
    }
)
xt_ag.lci()
xt_ag.lcia()

xt_others = reg.ExtensionTablesLCA(
    {('Sugarcane', 'driving'): 1},
    ('LC-IMPACT', 'Land Use', 'Occupation', 'Marginal', 'Core'),
    xtable='world-topo-ecoregions-weighted-pop-density',
    limitations={
        'activities': crops,
        'activities mode': 'exclude'
    }
)
xt_others.lci()
xt_others.lcia()

xt_ag.score + xt_others.score

xt_ag.fix_spatial_dictionaries()

def iterate_results_spatial_labels(matrix, axis, spatial_dict, cutoff=1e-4):
    _ = lambda x: x[1] if isinstance(x, tuple) else x
    
    rsd = {y: _(x) for x, y in xt_ag.ia_spatial_dict.items()}

    total = matrix.sum()
    summed = np.array(matrix.sum(axis=axis)).ravel()
    sorting = np.argsort(np.abs(summed))[::-1]

    summed = summed[sorting]
    mask = summed > cutoff * summed.max()

    for x, y in zip(summed[mask], sorting):
        yield x, x * 100 / total, rsd[y]

def to_geopandas(result_iter, geocollection):
    source = gpd.read_file(reg.geocollections[geocollection]['filepath'])
    merged = source.merge(pd.DataFrame(
        list(result_iter), 
        columns=['lcia_weight', 'lcia_weight_normalized', reg.geocollections[geocollection]['field']]
    ))
    return merged

df = to_geopandas(
    iterate_results_spatial_labels(
        (xt_ag.results_ia_spatial_scale() + xt_others.results_ia_spatial_scale()),
        0,
        xt_ag.ia_spatial_dict,
    ),
    'ecoregions'
)

m = folium.Map(location=[0, 0], zoom_start=2, 
               tiles="cartodbpositron")

df['geoid'] = df.index.astype(str)
geo_str = df.to_json()

m.choropleth(geo_str=geo_str,
             data=df, columns=['geoid', 'lcia_weight_normalized'],
             key_on='feature.id',
             fill_color='YlGn', fill_opacity=0.4, line_opacity=0.2)
m



