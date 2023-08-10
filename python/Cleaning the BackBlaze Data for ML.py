import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


### start program here

hdd_train = pd.read_csv('../input/harddrive.csv') #,nrows=50
hdd_test = pd.read_csv('../input/2016_jj.csv')

# merging of the test files resulted in multiple headers in the file -- remove them
hdd_test = hdd_test[hdd_test.smart_5_raw != 'smart_5_raw']
#print ("Number of test rows are {}".format(len(hdd_test.index)))

# removed normalized values, and model, and capacity, since they are constants
hdd_train = hdd_train.select(lambda x: x[-10:] != 'normalized', axis=1)
hdd_test = hdd_test.select(lambda x: x[-10:] != 'normalized', axis=1)

# found only in training file
hdd_train.drop(['smart_201_raw', 'smart_13_raw'], inplace=True, axis=1)

columns_to_drop =['date', 'capacity_bytes']  # new
hdd_train.drop(columns_to_drop, inplace=True, axis=1)
hdd_test.drop(columns_to_drop, inplace=True, axis=1)

# drop constant columns
hdd_train = hdd_train.loc[:, ~hdd_train.isnull().all()]
hdd_test = hdd_test.loc[:, ~hdd_test.isnull().all()]

hdd_train.columns

hdd_train.shape

hdd_test.columns

hdd_test.shape

# save the clean, new files
hdd_train.to_csv('../input/clean_train_data.csv', index=False)
hdd_test.to_csv('../input/clean_test_data.csv', index=False)

# remove strings from dataset
#columns_to_drop =['serial_number','model']
#hdd_train.drop(columns_to_drop, inplace=True, axis=1)
#hdd_train.fillna(-1, inplace=True)
#hdd_train = hdd_train.astype(np.float).fillna(-1.0)

#hdd_train.to_csv('../input/clean_train_data_nostr.csv', index=False)

hdd_train.head()

columns_to_use = ['serial_number', 'failure', 'smart_5_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw'
                 ]
# select specific model, since vendors differ on how SMART values are used
hdd = hdd_train.query('model == "ST4000DM000"')
hdd.shape

hdd.head()

hdd = hdd[columns_to_use]

# labelEnconde the serial_number
lbl = LabelEncoder()
lbl.fit(list(hdd['serial_number'].values))
hdd['serial_number'] = lbl.transform(list(hdd['serial_number'].values))
hdd.to_csv('../input/ST4000DM000_clean_SMART_harddrive.csv', index=False)
hdd.head()

hdd_fails_df = hdd[hdd.failure != 0]
hdd_fails_df.shape

hdd_fails_df.head()

hdd_fails_df = hdd_fails_df.drop_duplicates(['serial_number'], keep=False)

hdd_fails_df.to_csv('../input/failed_hdd_from_harddrive.csv', index=False)

