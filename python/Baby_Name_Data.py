import pandas as pd
import glob

filenames = glob.glob('names/yob*')

females = pd.DataFrame()
males = pd.DataFrame()

for filename in filenames:
    year = filename[9:13]
    data = pd.read_csv(filename, header=None, names=['Name','Gender',year], index_col='Name')
    females = females.join(data[data['Gender']=='F'].drop('Gender', axis=1), how='outer')
    males = males.join(data[data['Gender']=='M'].drop('Gender', axis=1), how='outer')

females.to_csv('female_names_timeseries.csv')
males.to_csv('male_names_timeseries.csv')

