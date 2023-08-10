with open('../vb_data/valencia_bisi_00000.json', 'r') as f:
    data = json.load(f)

data.keys()

df = pd.DataFrame(data)

len(df)

df.head()

df.reset_index(inplace=True)

df.drop(['index'], axis = 1)

'../vb_data/valencia_bisi_00'+"%03d" % (2,)

for i in range(0,1000):
    fname = '../vb_data/valencia_bisi_00'+"%03d" % (i,)+".json"
    
    with open(fname, 'r') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    df.reset_index(inplace = True)
    df.drop(['index'], axis =1)
        
    if i ==0:
        tot_data = df
        
    tot_data = tot_data.append(df)
tot_data.drop(['index'], axis =1)

len(tot_data)

tot_data.reset_index(inplace=True)

#tot_data.drop(['level_0'], axis =1, inplace = True)

tot_data

tot_data = tot_data[['update', 'X', 'Y', 'available', 'free', 'total', 'number', 'open', 'ticket', 'name']]

import utm

utm.to_latlon(725755.944, 4372972.613, 30, 'S')

tot_data.info()

tot_data['X'][0]

lat = []
long = []

for i in range(0,len(tot_data)):
    x = tot_data['X'][i]
    y = tot_data['Y'][i]
    lat.append(utm.to_latlon(x, y, 30, 'S')[1])
    long.append(utm.to_latlon(x, y, 30, 'S')[0])

#note that I switched these.
tot_data['Long'] = lat
tot_data['Lat'] = long

tot_data.drop(['X','Y'], axis =1, inplace= True) 
tot_data.to_csv('../vb_data/data.csv')

tot_data

