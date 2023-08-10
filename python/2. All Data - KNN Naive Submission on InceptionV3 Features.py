import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
from scipy.stats import mode

get_ipython().run_line_magic('matplotlib', 'inline')

train_df = pd.read_csv('../data/resized/train_resized.csv')
test_df = pd.read_csv('../data/resized/test_resized.csv')
sample_submission = pd.read_csv('./data/all/sample_submission.csv', usecols=['id'])

knn_distance = np.load('./result/knn_naive_distance.npy')
knn_neighbor = np.load('./result/knn_naive_neighbor_index.npy')

print('Train:\t\t\t', train_df.shape)
print('Test:\t\t\t', test_df.shape)
print('Sample Submission:\t', sample_submission.shape)
print('KNN Distance:\t\t', knn_distance.shape)
print('KNN Neighbor:\t\t', knn_neighbor.shape)

train_df.head()

test_df.head()

sample_submission.head()

# Get prediction for each query images
prediction = []
for neighbors in knn_neighbor:
    prediction.append(train_df.loc[neighbors[0]]['landmark_id'])

prediction_tuple = [str(idx) + ' ' + '1.0' for idx in prediction]

# Create submission files
submission = pd.DataFrame({'id': test_df['id'].values, 'landmarks': prediction_tuple})
submission = pd.merge(sample_submission, submission, how='left', on='id')
submission.to_csv('./result/knn_naive_first_neighbor.csv', index=False, columns=['id', 'landmarks'])

# Get the first 100 neighbors
predictions = []
for neighbors in knn_neighbor:
    predictions.append(train_df.loc[neighbors]['landmark_id'].values)

predictions = np.array(predictions)

# Get mode
prediction_mode = mode(predictions, axis=1)
prediction = prediction_mode[0][:, 0]
prediction_tuple = [str(idx) + ' ' + '1.0' for idx in prediction]

# Create submission files
submission = pd.DataFrame({'id': test_df['id'].values, 'landmarks': prediction_tuple})
submission = pd.merge(sample_submission, submission, how='left', on='id')
submission.to_csv('./result/knn_naive_mode_neighbor.csv', index=False, columns=['id', 'landmarks'])



