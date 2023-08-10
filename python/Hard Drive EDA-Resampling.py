import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import ensemble, metrics
from IPython.display import Image
from IPython.core.display import HTML 

# limit to first 1000 rows for now until doc is complete
hdd = pd.read_csv('../input/harddrive.csv')#,nrows = 10000)
hdd.head()

# number of rows and columns in dataset
hdd.shape

# number of hdd
hdd['serial_number'].value_counts().shape

# number of different model types of harddrives
hdd['model'].value_counts().shape

#failed drives (model and count)
print(hdd.groupby('model')['failure'].sum().sort_values(ascending=False).iloc[:30])

'''
#columns_to_drop =['date','smart_1_normalized', 'smart_1_raw', 
#'smart_2_normalized', 'smart_2_raw',
#'smart_3_normalized', 'smart_3_raw', 'smart_4_normalized', 'smart_4_raw',
#'smart_7_normalized', 'smart_7_raw', 'smart_8_normalized', 'smart_8_raw', 
#'smart_9_normalized', 'smart_9_raw', 'smart_13_normalized', 'smart_13_raw',
#'smart_190_normalized', 'smart_190_raw', 'smart_191_normalized', 'smart_191_raw',
#'smart_192_normalized', 'smart_192_raw', 'smart_193_normalized', 'smart_193_raw', 
#'smart_194_normalized', 'smart_194_raw', 'smart_195_normalized', 'smart_195_raw', 
#'smart_199_normalized', 'smart_199_raw', 'smart_200_normalized', 'smart_200_raw',
#'smart_220_normalized', 'smart_220_raw', 'smart_222_normalized', 'smart_222_raw',
#'smart_223_normalized', 'smart_223_raw', 'smart_224_normalized', 'smart_224_raw', 
#'smart_225_normalized', 'smart_225_raw', 'smart_226_normalized', 'smart_226_raw', 
#'smart_240_normalized', 'smart_240_raw', 'smart_241_normalized', 'smart_241_raw', 
#'smart_242_normalized', 'smart_242_raw', 'smart_250_normalized', 'smart_250_raw', 
#'smart_251_normalized', 'smart_251_raw', 'smart_252_normalized', 'smart_252_raw', 
#'smart_254_normalized', 'smart_254_raw', 'smart_255_normalized', 'smart_255_raw']
'''
columns_to_drop =['date', 'capacity_bytes']
hdd.drop(columns_to_drop, inplace=True, axis=1)

# drop constant columns
hdd = hdd.loc[:, ~hdd.isnull().all()]
# remove normalized values that are left
#hdd = hdd.select(lambda x: x[-10:] != 'normalized', axis=1)

# no null values left. 
hdd.isnull().any()
hdd.fillna(-1, inplace=True)
#hdd = hdd.drop(['model', 'capacity_bytes'], axis=1)
hdd.head()

hdd.shape

hdd.columns

# select specific model, since vendors differ on how SMART values are used
hdd = hdd.query('model == "ST4000DM000"')
hdd.shape

from sklearn.preprocessing import LabelEncoder
serial_encoder = LabelEncoder()
hdd['serial_number'] = serial_encoder.fit_transform(hdd['serial_number'].astype('str'))
hdd.head()

# number of unique hdd
serials_df = pd.DataFrame()
serials_df['serial_number'] = hdd['serial_number']
serials_df.drop_duplicates('serial_number', inplace=True)
print len(hdd['serial_number'].unique())
serials_df.shape[0]

#number of failed drives
print hdd.loc[hdd['failure'] == 1].shape[0]

# remove normalized values that are left
hdd = hdd.select(lambda x: x[-10:] != 'normalized', axis=1)
hdd.shape

# remove model number
hdd.drop(['model'], inplace=True, axis=1)

from imblearn.over_sampling import SMOTE
# Apply SMOTE's
X = hdd
y = np.asarray(hdd['failure'])
kind = 'regular'
sm = SMOTE(kind='regular')
#X.shape
X_res, y_res = sm.fit_sample(X, y)
X_res.shape

#save solution set
serial_df = pd.DataFrame()
serial_df['serial_number'] = hdd['serial_number']
serial_df['failure'] = hdd['failure']
serial_df.head()

serial_df.shape

hdd.columns

# save the clean, new files
hdd.to_csv('../input/harddrive_resampled.csv', index=False)
serial_df.to_csv('../input/solutions_resampled.csv', index=False)

