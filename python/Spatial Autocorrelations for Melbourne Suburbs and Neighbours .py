import numpy as np
import pandas as pd
from pandas import Series,DataFrame

dframe= pd.read_csv('data/spatial_autocorrelation.csv')
dframe.columns = dframe.columns.str.strip()

dframe = dframe.drop(['Walkability Index_Scaled','Walkability Index_Lagged_Scaled'],axis=1)

dframe['Diff'] = abs(dframe['Walkability Index'].sub(dframe['Walkability Index_Lagged'],axis=0))

dframe.sort_values('Diff',ascending=False)

