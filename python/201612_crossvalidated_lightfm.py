import lightfm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

from lightfm.datasets import fetch_stackexchange

data = fetch_stackexchange('crossvalidated',
                           test_set_fraction=0.1,
                           indicator_features=False,
                           tag_features=True)

data['test_positive_only'] = data['test'].copy()
train = data['train']
test = data['test']
test_positives = data['test_positive_only']

print(np.unique(data['train'].data))
print(data['train'].__repr__())
print(np.unique(data['test'].data))
print(data['test'].__repr__())
print(np.unique(data['test_positive_only'].data))
print(data['test_positive_only'].__repr__())

item_features = data['item_features']
tag_labels = data['item_feature_labels']
print('There are %s distinct item features, with values like %s.' % (item_features.shape[1], tag_labels[:3].tolist()))

from lightfm.datasets import fetch_movielens

data = fetch_movielens('movielens', indicator_features=False, genre_features=True)

print('original train')
print(np.unique(data['train'].data))
print(data['train'].__repr__())
print('original test')
print(np.unique(data['test'].data))
print(data['test'].__repr__())

# binarizing traing examples as in the original lightfm paper to use the logistic loss
data['train'].data = np.array([-1, 1])[1 * (data['train'].data >= 4)]
data['test'].data = np.array([-1, 1])[1 * (data['test'].data >= 4)]

# should keep only positive test interactions
data['test_positive_only'] = data['test'].copy()
data['test_positive_only'].data = 1 *(data['test_positive_only'].data>=1)
data['test_positive_only'].eliminate_zeros()

train = data['train']
test = data['test']
test_positives = data['test_positive_only']

print('train')
print(np.unique(data['train'].data))
print(data['train'].__repr__())
print('test')
print(np.unique(data['test'].data))
print(data['test'].__repr__())
print('test_positive_only')
print(np.unique(data['test_positive_only'].data))
print(data['test_positive_only'].__repr__())

item_features = data['item_features']
tag_labels = data['item_feature_labels']
print('There are %s distinct item features, with values like %s.' % (item_features.shape[1], tag_labels[:3].tolist()))

from lightfm import LightFM

NUM_EPOCHS = 3
cf_model = LightFM(
    loss='warp',
    item_alpha=1e-6,
    no_components=10)

get_ipython().magic('time model = cf_model.fit(train, epochs=NUM_EPOCHS)')

def sigmoid(x):                                        
    return 1 / (1 + np.exp(-x))
    
def log_loss(df):
    return -((df.rating == 1) * np.log(df.predicted_rating) + (df.rating == -1) * np.log(1 - df.predicted_rating))

def predicted_df(mat, predicted_values):
    return pd.DataFrame.from_dict({
        'user': mat.row,
        'item': mat.col,
        'rating': mat.data,
        'predicted_rating': predicted_values})\
    .sort_values('user')\
    .assign(log_loss=log_loss)

def model_log_losses_df(model, test_mat, item_features=data['item_features']):
    return predicted_df(
        test_mat, 
        sigmoid(cf_model.predict(test_mat.row, test_mat.col, item_features=item_features)))

predicted_train_df = predicted_df(train, sigmoid(cf_model.predict(train.row, train.col)))
print(predicted_train_df.log_loss.mean())
print(predicted_train_df.shape)

predicted_test_df = predicted_df(test, sigmoid(cf_model.predict(test.row, test.col)))
print(predicted_test_df.log_loss.mean())
print(predicted_test_df.shape)
print(model_log_losses_df(cf_model, test_positives).log_loss.mean())
predicted_test_df.head()

from lightfm.evaluation import reciprocal_rank, auc_score

cf_model = LightFM(loss='logistic', item_alpha=1e-6, no_components=30)
cf_model.fit(train, epochs=NUM_EPOCHS)

print('Collaborative filtering train/test MRR: %.3f / %.3f'
      % (reciprocal_rank(cf_model, train).mean(),
         reciprocal_rank(cf_model, test).mean()))

print('Collaborative filtering train/test AUC: %.3f / %.3f'
      % (auc_score(cf_model, train).mean(),
         auc_score(cf_model, test, train_interactions=None).mean()))

def sparse_to_df(mat):
    coo_mat = mat.tocoo()
    return pd.DataFrame({
        'user': coo_mat.row,
        'item': coo_mat.col,
        'rank': coo_mat.data}        
    )

def model_to_mrr(model, test_mat):
    predict_kwargs = {}
    if data['item_features'].shape[1] == model.item_embeddings.shape[0]:
        predict_kwargs['item_features'] = data['item_features']
    #if data['user_features'].shape[1] == model.user_embeddings.shape[0]:
    #    predict_kwargs['user_features'] = data['user_features']
        
    return sparse_to_df(model.predict_rank(test_mat, **predict_kwargs))        .assign(rec_rank=lambda df:1 / (df['rank'] + 1))        .groupby('user')['rec_rank'].max()        .mean()

test_ranks = cf_model.predict_rank(test)
test_predicted_ranks_df = sparse_to_df(test_ranks)
print(test_predicted_ranks_df.shape)
print(model_to_mrr(cf_model, test))
test_predicted_ranks_df.head()

(np.sum(test_ranks.sum(axis=1).A > 0), np.sum(test_ranks.sum(axis=0).A > 0)), test_ranks.shape, test_ranks.nnz

test_ranks

model = LightFM(loss='warp',
                item_alpha=1e-6,
                no_components=30)

model = model.fit(
    data['train'],
    item_features=data['item_features'],
    epochs=3)

print(model_log_losses_df(model, train).log_loss.mean())
print(model_log_losses_df(model, test).log_loss.mean())
print(model_log_losses_df(model, test_positives).log_loss.mean())

from lightfm.evaluation import reciprocal_rank, auc_score

print('Collaborative filtering train/test MRR: %.3f / %.3f'
      % (reciprocal_rank(model, train, item_features=data['item_features']).mean(),
         reciprocal_rank(model, test, item_features=data['item_features']).mean()))

print('Collaborative filtering train/test AUC: %.3f / %.3f'
      % (auc_score(model, train, item_features=data['item_features']).mean(),
         auc_score(model, test, item_features=data['item_features'], train_interactions=train).mean()))

