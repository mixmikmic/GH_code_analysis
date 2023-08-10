import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

blocks_map = pd.read_csv('census_block_loc.csv')
census = pd.read_csv('nyc_census_tracts.csv', index_col=0)

blocks_map.head()

census.head()

census.County.unique()

blocks_map = blocks_map[blocks_map.County.isin(['Bronx','Kings','New York','Queens', 'Richmond'])]
blocks_map['Tract'] = blocks_map.BlockCode // 10000
blocks_map['Tract'] = blocks_map.BlockCode // 10000
blocks_map = blocks_map.merge(census,how='left',right_index=True,left_on='Tract')

blocks_map.tail()

def convert_to_2d(lats,lons,values):
    latmin = 40.48
    lonmin = -74.28
    latmax = 40.93
    lonmax = -73.65
    lon_vals = np.mgrid[lonmin:lonmax:200j]
    lat_vals = np.mgrid[latmin:latmax:200j]
    map_values = np.zeros([200,200])
    dlat = lat_vals[1] - lat_vals[0]
    dlon = lon_vals[1] - lon_vals[0]
    for lat,lon,value in zip(lats,lons,values):
        lat_idx = int(np.rint((lat - latmin) / dlat))
        lon_idx = int(np.rint((lon-lonmin) / dlon ))        
        if not np.isnan(value):
            map_values[lon_idx,lat_idx] = value
    return lat_vals,lon_vals,map_values

def make_plot(blocks, data_values,title='',colors='Greens'):
    lat_vals,lon_vals,values = convert_to_2d(blocks.Latitude,blocks.Longitude,data_values)
    fig, ax = plt.subplots(figsize = [12,12])
    #fig = plt.figure(1,figsize=[10,10])
    limits = np.min(lon_vals),np.max(lon_vals),np.min(lat_vals),np.max(lat_vals)
    
    im = ax.imshow(values.T,origin='lower',cmap=colors,extent=limits, zorder = 1)
    ax.autoscale(False)
    plt.xlabel('Longitude [degrees]')
    plt.ylabel('Latitude [degrees]')
    plt.title(title)
    plt.colorbar(im,fraction=0.035, pad=0.04)
    plt.show()

make_plot(blocks_map, blocks_map.Income,colors='plasma',title='Median Household Income ($)')
# make_plot(blocks_map, blocks_map.IncomePerCap,colors='plasma',title='Per Capita Income ($)')

make_plot(blocks_map, blocks_map.Transit,colors='plasma',title='Percentage Taking Public Transportation to Work')

make_plot(blocks_map, blocks_map.Women,colors='plasma',title='Female Population')

blocks_map[['Latitude','Longitude', 'BlockCode', 'County_x', 'Women', 'Men', 'TotalPop']].sort_values('Women',ascending=False).head(35)

blocks_map.Women.describe()

filtered_blocks = blocks_map[['Latitude', 'Longitude', 'BlockCode', 'County_x', 'Women', 'TotalPop', 'Transit', 'Income']]

filtered_blocks.info()

filtered_blocks['women_pop_perc'] = np.divide(np.array(filtered_blocks['Women'], int),np.array(filtered_blocks['TotalPop'],int))

filtered_blocks.women_pop_perc.describe()

filtered_blocks.Transit.describe()

#filtered_blocks = filtered_blocks[filtered_blocks.women_pop_perc > 0.51]

#filtered_blocks = filtered_blocks[filtered_blocks.Income >150000]

#filtered_blocks = filtered_blocks[filtered_blocks.Transit > 40]

filtered_blocks.describe()

filtered_blocks

top_stations = [['34 St - Penn Station',40.7520, -73.9933], 
                ['34 St - Herald Sq',40.7496, -73.9877], 
                ['Grand Central 42nd St',40.7527, -73.9772] , 
                ['Times Sq 42nd St',40.7553, -73.9869],
                ['23 St', 40.7427, -73.9926],
                ['86 St',40.4640,-73.5706] ,
                ['Fulton St', 40.7094, 74.0083], 
                ['59 St', 40.7684, -73.9818],
                ['125 St',40.8049, 73.9385],
                ['14 St-Union Sq',40.4429,-73.5914],
                ['47-50 Sts Rock',40.7587, 73.9813],
                ['Flushing-Main', 40.7584, 73.8305] ,
                ['96 St',40.7942, 73.9721],
                ['42 St-Port Auth',40.4525, -73.5923],
                ['14 St',40.7388, -73.9997], 
                ['59 St Columbus', 40.4659, -73.5854],
                ['Canal St', 40.4373, -74.0036],
                ['50 St', 40.761, -73.9840],
                ['72 St', 40.468, -73.5730], 
                ['28 St',40.7433, 73.9841]]

top_stations_df = pd.DataFrame(data = top_stations, columns = ['Station', 'Lat','Long'])

top_stations_df



filtered_blocks[((filtered_blocks['Latitude'] < 40.753) & (filtered_blocks['Latitude'] > 40.751))&((filtered_blocks['Longitude']>-73.995)&(filtered_blocks['Longitude']<-73.991))]#.describe()

filtered_blocks[((filtered_blocks['Latitude'] < 40.750) & (filtered_blocks['Latitude'] > 40.747))&((filtered_blocks['Longitude']>-73.986)&(filtered_blocks['Longitude']<-73.984))]

filtered_blocks[((filtered_blocks['Latitude'] < 40.754) & (filtered_blocks['Latitude'] > 40.752))&((filtered_blocks['Longitude']>-73.977)&(filtered_blocks['Longitude']<-73.976))]

filtered_blocks[((filtered_blocks['Latitude'] < 40.756) & (filtered_blocks['Latitude'] > 40.754))&((filtered_blocks['Longitude']>-73.989)&(filtered_blocks['Longitude']<-73.984))]



