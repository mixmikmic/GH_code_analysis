import geopandas as gpd
import pandas as pd
import pickle

# load, clean, and normalize country-level lights data
with open('data/geo/pickles/zonal_stats_c.pickle') as f:
    gdf = pickle.load(f)
gdf = pd.DataFrame(gdf)
gdf = gdf.drop_duplicates(subset='WB_A3')
gdf = gdf.set_index('WB_A3')
gdf.drop(['ADMIN', 'CONTINENT', 'ISO_A3', 'REGION_UN', 'REGION_WB', 'SUBREGION', 'geometry'], axis=1, inplace=True)
gdf_normalizer = (gdf.F182013).as_matrix()
gdf_normed = gdf.divide(gdf_normalizer, axis=0)

# Load, clean, and normalize wb data
wb = pd.read_csv('data/econ/wb.csv')
label = 'GDP, PPP (constant 2011 international $)'
wb = wb[wb['Series Name'] == label]
wb.drop(['Country Name', 'Series Name', 'Series Code', '2014', '2015'], axis=1, inplace=True)
wb.rename(columns={'Country Code': 'WB_A3'}, inplace=True)
#wb.dropna(axis=0, inplace=True)
wb = wb.set_index('WB_A3')
wb_normalizer = (wb['2013']).as_matrix()
wb_normed = wb.divide(wb_normalizer, axis=0)

# join lights and wb datasets
df = gdf_normed.join(wb_normed, how='inner')

# pickle joined dataframe
df.to_csv('data/geo/zonal_stats_c_norm.csv')
df.to_pickle('data/geo/pickles/zonal_stats_c_norm.pickle')
wb.to_pickle('data/geo/pickles/wb_data.pickle')

# syria case: normalize using 2007 gdp data in constant 2005 usd
gdf_syria = gdf.loc['SYR'].to_frame().transpose()
gdf_syria_normalizer = (gdf_syria.F162007).as_matrix()
gdf_syria_normed = gdf_syria.divide(gdf_syria_normalizer, axis=0)

wb_syria = pd.read_csv('data/econ/wb.csv')
label = 'GDP at market prices (constant 2005 US$)'
wb_syria = wb_syria[wb_syria['Series Name'] == label]
wb_syria.drop(['Country Name', 'Series Name', 'Series Code', '2014', '2015'], axis=1, inplace=True)
wb_syria.rename(columns={'Country Code': 'WB_A3'}, inplace=True)
wb_syria = wb_syria.set_index('WB_A3')
wb_syria = wb_syria.loc['SYR'].to_frame().transpose(); wb_syria
wb_syria_normalizer = (wb_syria['2007']).as_matrix()
wb_syria_normed = wb_syria.divide(wb_syria_normalizer, axis=0)

# join lights and wb datasets
df_syria = gdf_syria_normed.join(wb_syria_normed, how='inner')

# pickle joined dataframe
df_syria.to_csv('data/geo/zonal_stats_c_norm_syr.csv')
df_syria.to_pickle('data/geo/pickles/zonal_stats_c_norm_syr.pickle')
wb_syria.to_pickle('data/geo/pickles/wb_data_syr.pickle')

# angola and south sudan case: normalize using 2013 gdp data in current usd
gdf_agossd = gdf.loc[['SSD', 'AGO']]
gdf_agossd_normalizer = (gdf_agossd.F182013).as_matrix()
gdf_agossd_normed = gdf_agossd.divide(gdf_agossd_normalizer, axis=0)

wb_agossd = pd.read_csv('data/econ/wb.csv')
label = 'GDP at market prices (current US$)'
wb_agossd = wb_agossd[wb_agossd['Series Name'] == label]
wb_agossd.drop(['Country Name', 'Series Name', 'Series Code', '2014', '2015'], axis=1, inplace=True)
wb_agossd.rename(columns={'Country Code': 'WB_A3'}, inplace=True)
wb_agossd = wb_agossd.set_index('WB_A3')
wb_agossd = wb_agossd.loc[['SSD', 'AGO']]
wb_agossd_normalizer = (wb_agossd['2013']).as_matrix()
wb_agossd_normed = wb_agossd.divide(wb_agossd_normalizer, axis=0)

# join lights and wb datasets
df_agossd = gdf_agossd_normed.join(wb_agossd_normed, how='inner')

# pickle joined dataframe
df_agossd.to_pickle('data/geo/pickles/zonal_stats_c_norm_agossd.pickle')
wb_agossd.to_pickle('data/geo/pickles/wb_data_agossd.pickle')

