import pandas as pd
import os

csvs = [files for files in os.listdir('./') if files.endswith('.log')]

pd.read_csv(csvs[1]).head(5)

von = 6.0
bis = 14.0

values = ['accelerationX','accelerationY','accelerationZ','motionRotationRateX','motionRotationRateY', 'motionRotationRateZ']

# Write Header in CSV
with open('activitydata.csv', 'wb') as f:
    f.write("timestamp," + ",".join(values) + ',activity\n')

# Open all Activity csvs and get data
for csv in csvs:
    print('Lade %s' % csv)
    data = pd.read_csv(csv, index_col='recordtime')
    data = data[(data.index<=bis) & (data.index>=von)] # equal time ranges
    
    data.index = data.time
    data = data[values] # just keep relevant values

    data['activity'] = csv.split('_')[1] # label data
    
    with open('activitydata.csv', 'a') as f:
        data.to_csv(f, header=False, float_format='%.6f')



