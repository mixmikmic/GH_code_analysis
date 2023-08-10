import numpy as np
import pandas as pd
import folium
from folium import plugins
import pandas as pd
import MySQLdb
from sqlalchemy import create_engine
engine = create_engine('mysql://root:rektroot123@dbprojectaws.csc6seakwleg.us-east-1.rds.amazonaws.com/dbprojectAWS')

selected_coords = (41.8897, -87.6222)

coords_file = pd.read_sql_query("SELECT * from venues where rating_count>0;", engine)
coords = coords_file[['lat','lon','normalised_rating']]
coords1 = coords.values.tolist()

from functools import partial
from heapq import nsmallest
dist=lambda s,d: (s[0]-d[0])**2+(s[1]-d[1])**2
a = coords1
closest_coords_list_1 = nsmallest(17, a, key=partial(dist, selected_coords))
def fuse(points, d, rat):
    ret = []
    d2 = d * d
    n = len(points)
    taken = [False] * n
    for i in range(n):
        if not taken[i]:
            count = 1
            point = [points[i][0], points[i][1]]
            taken[i] = True
            for j in range(i+1, n):
                if dist(points[i], points[j]) < d2:
                    point[0] += points[j][0]
                    point[1] += points[j][1]
                    count+=1
                    taken[j] = True
            point[0] /= count
            point[1] /= count
            ret.append((point[0], point[1], np.mean(rat[i])))
    return ret
closest_coords_list = fuse(closest_coords_list_1, 0.0001, [item[2] for item in closest_coords_list_1])

closest_coords = sorted(closest_coords_list,key=lambda x: (x[2]), reverse = True)
closest_coords_array = np.asarray(closest_coords)
closest_coords_df = pd.DataFrame(closest_coords_array)
closest_coords_df.rename(columns={
                 0: 'Lat',
                 1: 'Lon',
                 2: 'Normalised_Rating'}, inplace=True)

m = folium.Map([selected_coords[0], selected_coords[1]], zoom_start=18)
for index, row in closest_coords_df.iterrows():
    folium.Marker([row['Lat'], row['Lon']],
                       popup='{}'.format(row['Normalised_Rating'])).add_to(m)
m

