import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn
get_ipython().magic('matplotlib inline')
seaborn.set()

school_data = pd.read_csv("../data/Directory-School-current.csv")

school_data.columns

school_data.ix[:, ['Name', 'Longitude ', 'Latitude']].to_csv('../data/school_data.csv')

school_data.ix[:, ['Name', 'Longitude ', 'Latitude']]    .apply(lambda x: ~np.any(x.isnull()), axis = 1)    .pipe(lambda x: school_data.ix[x, ['Name', 'Longitude ', 'Latitude']])    .to_csv('../data/school_data.csv', index = False)

get_ipython().magic('pinfo school_data.apply')

