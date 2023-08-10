import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')

train_feature = np.load('./data/resized/train_features.npy')
train_df = pd.read_csv('./data/resized/train_resized.csv')

train_labels = train_df['landmark_id'].values

# Choose the unique ids
unique_ids = sorted(train_df['landmark_id'].unique())
len(unique_ids)

# Group data according unique landmark_id
grouped = train_df[['landmark_id', 'id']].groupby('landmark_id').count().reset_index()
grouped = grouped.sort_values('id', ascending=False)
grouped = grouped.rename(columns={'id': 'count'}).reset_index(drop=True)

# About 41% landmark ids have image less than 10
len(grouped[grouped['count'] < 10]) / len(grouped)

# Split into training and test set
train_ids = []
test_ids = []

for idx in unique_ids:
    index = list(train_df[train_df['landmark_id'] == idx].index)
    np.random.shuffle(index)
    
    if len(index) >= 12:
        train_ids += index[:10]
        test_ids += index[10:12]
    elif len(index) >= 10:
        train_ids += index[:10]
        test_ids += index[10:]
    elif len(index) >= 3:
        train_ids += index[:-1]
        test_ids.append(index[-1]) 
    else:
        train_ids += index

print('New train:\t', len(train_ids))
print('New test:\t', len(test_ids))

# Select training and testing subsets
sub_train_df = train_df.loc[train_ids]
sub_test_df = train_df.loc[test_ids]

sub_train_feature = train_feature[train_ids]
sub_test_feature = train_feature[test_ids]

# Save to disk ./data/subset
sub_train_df.to_csv('./data/knn/train.csv', index=False)
sub_test_df.to_csv('./data/knn/test.csv', index=False)

np.save('./data/knn/train_feature.npy', sub_train_feature)
np.save('./data/knn/test_feature.npy', sub_test_feature)

