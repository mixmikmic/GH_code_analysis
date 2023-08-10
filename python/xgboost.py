import sys
sys.path.append('../')
sys.path.append('../support/')
from ct_reader import *
from glob import glob
import timeit
from os.path import join, basename, isfile
from tqdm import tqdm
from functools import partial
from paths import *
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
import pandas as pd
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import scale
import xgboost as xgb
from sklearn import cross_validation
import pickle
get_ipython().magic('pylab inline')

def to_dataframe(stats):
    columns = ['id', 'max', 
               'amoun', 'mean', 
               'median_not_min', 
               'mean_not_min', 
               'std', 'areas', 
               'median'] + ['modes_' + str(i) 
                            for i in range(9)]

    df = pd.DataFrame(None, columns=columns)
    
    for isolated in tqdm(stats):
        tmp = dict()
        if 'modes' in  isolated[1].keys():
            isolated[1]['modes'] = [sum(threshold)
                                    for threshold in isolated[1]['modes']]
        else: 
            isolated[1]['modes'] = [0] * 9
            
        for i in range(9):
            tmp['modes_' + str(i)] = [isolated[1]['modes'][i]]
        tmp['id'] = isolated[0]
        tmp['areas'] = [sum(isolated[1]['areas'])]
        remind = set(isolated_stats[0][1].keys())
        remind = remind.difference(['modes', 'areas'])
        for key in remind:
            tmp[key] = [isolated[1][key]]
        df = df.append(pd.DataFrame(tmp))
    return df

def extract_enhs(df_features, df_labels=None):
    if df_labels is not None:
        df_features['cancer'] = NaN
        for row in df_labels.iterrows():
            df_features.loc[df_features['id'] == row[1]['id'], 'cancer'] = row[1].cancer
        df_features.dropna(inplace=True)
        return df_features.drop(['cancer', 'id'], axis=1), df_features.cancer
    else:
        return df_features.drop(['id'], axis=1), df_features['id']

def extract_mxnet(df=None, mxnet='MAX_TOP', pca=False):
    x = [[load(join(PATH['DATA_MXNET'], mxnet, 
                    '%s.npy' % str(patient))).reshape(-1, 2048)] 
         for patient in df['id'].tolist()]

    coords = [0]
    for el in x:
        coords.append(el[0].shape[0] + coords[-1])

    x = vstack([el[0] for el in x])

    features = list()
    for i in range(1, len(coords)):
        features.append(mean(x[coords[i - 1]: coords[i]], axis=0))

    features = vstack(features)
    if pca:
        features = PCA(pca).fit_transform(features)
    
    x = [load(join(PATH['DATA_ENHANCED'], 
                    '%samounts.npy' % str(patient))) 
         for patient in df['id'].tolist()]
    features = hstack([features, x])
    
    if 'cancer' in df.columns:
        return features, df['cancer']
    else:
        return features

def train_svc(features=None, y=None, 
              tv_split=.2):
    
    features = scale(features)
    trn_x, val_x, trn_y, val_y = train_test_split(features, y, 
                                                  random_state=42, 
                                                  stratify=y,
                                                  test_size=tv_split)

    print(trn_x.shape, trn_y.shape)
    
    clf = LinearSVC(C=1, class_weight={0: 0.67487923, 
                                          1: 1.92955801})

    clf.fit(trn_x, trn_y)
    print(log_loss(val_y, clf.predict(val_x)))
    return clf

def train_xgboost(features=None, y=None, cross_val=False):
        
    trn_x, val_x, trn_y, val_y = train_test_split(features, y, 
                                                  random_state=42, 
                                                  stratify=y,
                                                  test_size=.2)
    

    if cross_val:
        kfold = cross_validation.StratifiedKFold(train.cancer, 
                                                 n_folds=5, 
                                                 shuffle=True,
                                                 random_state=42)
        folds = kfold.test_folds
        for i in range(kfold.n_folds):
            trn_x = features[folds != i]
            val_x = features[folds == i]
            trn_y = y[folds != i]
            val_y = y[folds == i]
            
#             clf = xgb.XGBRegressor(max_depth=10,
#                             n_estimators=20000,
# #                             min_child_weight=9,
#                             learning_rate=0.001,
# #                             max_delta_step=1,
#                             nthread=8,
#                             subsample=0.80,
#                             colsample_bytree=0.90,
#                             seed=4242)

            clf = xgb.XGBRegressor(max_depth=13,
                            n_estimators=20000,
#                             min_child_weight=9,
                            learning_rate=0.003,
#                             max_delta_step=1,
                            nthread=8,
                            subsample=0.80,
                            colsample_bytree=0.90,
                            seed=4242)
            
            clf.fit(trn_x, trn_y, 
                    eval_set=[(val_x, val_y)], 
                    verbose=200, 
                    eval_metric='logloss', 
                    early_stopping_rounds=400)
    return clf

df_mxnet = pd.read_csv(join(PATH['CSV'], 'stage1_labels.csv'))
features, y = extract_mxnet(df_mxnet, 'MAX_FRONT', False)

def add_modes(df, modes, prefix):
    modes_new = zeros(shape=(len(modes), 9))
    for i in range(9):
        for j in range(len(modes)):
            modes_new[j][i] = int(modes[j][1:-1].split(', ')[i])
    for i in range(9):
        df[prefix + '_' + str(i)] = modes_new[:, i]
    return df

TOP_AMOUNT = 10
DENCE_AMOUNT = 256

predicted = glob(join(PATH['DATA_OUT'], 'DATAFRAMES', 'predict*.csv'))
df = list()
for path in tqdm(predicted):
    df.append(pd.read_csv(path))
df = pd.concat(df)

df['patientid'] = df.patchid.apply(lambda x: x[:32])
df = df.sort_values(['probability'], ascending=[False])
grouped = df.groupby('patientid', sort=False)
stats = df.groupby('patientid', sort=False).agg({'probability': sum})
rrrrr = df.groupby('patientid', sort=False).agg({'probability': mean})
stats['patientid'] = stats.index
rrrrr['patientid'] = rrrrr.index
stats = rrrrr.merge(stats, on='patientid')

top = grouped.head(10).reset_index(drop=True)
top_patches = top.patchid
top_patches.to_csv(join(PATH['DATA_OUT'], 'DATAFRAMES', 'top_patches'), index=False, header=['patchid'])
top = top.sort_values(['patientid'], ascending=[False])

columns = ['patientid']         + ['probability_' + str(i) for i in range(TOP_AMOUNT)]        + ['dence_sumx_' + str(i) for i in range(DENCE_AMOUNT)]        + ['dence_sumy_' + str(i) for i in range(TOP_AMOUNT)]
data = list()
bads = list()

for name, group in tqdm(grouped):
    flag = False
    row = dict()
    dences = list()
    for ndence, nprob in zip(group[:10].dence, group[:10].probability):
        dences.append([float(part) * nprob for part in 
                 ndence
                 .replace('\n', ' ')
                 .replace('[', ' ')
                 .replace(']', ' ')
                 .replace('"', ' ')
                 .split(' ') 
                 if part])
    
        if len(dences[-1]) != DENCE_AMOUNT:
            flag = True
            
    if flag:
        bads.append((name, group))
        continue
            
    if len(dences) != TOP_AMOUNT:
        bads.append((name, group))
        continue
    
    dence_sumy = asarray(dences).sum(1) / group[:10].probability.sum()
    dence_sumx = asarray(dences).sum(0) / group[:10].probability.sum()
        
    for i in range(TOP_AMOUNT):
        row['probability_' + str(i)] = [group[:10].probability.values[i]]
    for i in range(TOP_AMOUNT):
        row['dence_sumy_' + str(i)] = [dence_sumy[i]]
    for i in range(DENCE_AMOUNT):
        row['dence_sumx_' + str(i)] = [dence_sumx[i]]
    row['patientid'] = [group.patientid.values[0]]
    
    data.append(pd.DataFrame(row))
    
data = pd.concat(data)
# data.to_csv(join(PATH['DATA_OUT'], 'DATAFRAMES', 'data_wo_280'))

train = data.merge(pd.read_csv(PATH['LABELS']), 
         left_on='patientid', right_on='id')

clf = train_xgboost(train.drop(['id', 'patientid', 'cancer'], axis=1), train.cancer, cross_val=True)

df_sample = pd.read_csv(PATH['SAMPLE'])   
df = pd.read_csv(join(PATH['DATA_OUT'],
                      'DATAFRAMES',
                      'stats_not_full.csv')).drop(['Unnamed: 0'], axis=1)

test = df.merge(df_sample.drop(['cancer'], axis=1), 
                  left_on='id', right_on='id')
    
test = add_modes(test.drop(['iso_modes'], axis=1), 
                 test.iso_modes.values, 'iso_modes_')
test = add_modes(test.drop(['vas_modes'], axis=1), 
                 test.vas_modes.values, 'vas_modes_')
test = add_modes(test.drop(['plu_modes'], axis=1), 
                 test.plu_modes.values, 'plu_modes_')
test_solutions = pd.read_csv(join(PATH['CSV'], 'stage1_solution.csv'))
test = test.merge(test_solutions.drop(['Usage'], axis=1), 
           left_on='id', 
           right_on='id')

df = pd.read_csv(join(PATH['DATA_OUT'],
                      'DATAFRAMES',
                      'stats_not_full.csv')).drop(['Unnamed: 0'], axis=1)
train = df.merge(pd.read_csv(PATH['LABELS']), 
         left_on='id', right_on='id')

train = add_modes(train.drop(['iso_modes'], axis=1), train.iso_modes.values, 'iso_modes_')
train = add_modes(train.drop(['vas_modes'], axis=1), train.vas_modes.values, 'vas_modes_')
train = add_modes(train.drop(['plu_modes'], axis=1), train.plu_modes.values, 'plu_modes_')

train = pd.concat([train, test])

clf = train_xgboost(df[df.cancer != -1].drop(['id', 'cancer'], axis=1), train.cancer, cross_val=True)

train.columns

test.drop(['id', 'cancer'], axis=1).columns == df[df.cancer != -1].drop(['id', 'cancer'], axis=1).columns

df_sample = pd.read_csv(join(PATH['CSV'], 'stage2_sample_submission.csv'))   

test = df.merge(df_sample.drop(['cancer'], axis=1), 
                left_on='id', right_on='id')

pred = clf.predict(test.drop(['id',  'cancer'], axis=1))

for idx, pred_value in zip(test['id'], pred):
    df_sample.loc[df_sample['id'] == idx, 'cancer'] = pred_value

ids = df_sample[df_sample.cancer == .5]['id'].values
old = pd.read_csv(join(PATH['CSV'], 'last_best.csv'))
for idx in ids:
    if len(old[old['id'] == idx]):
        df_sample.loc[df_sample['id'] == idx, 'cancer'] = old[old['id'] == idx].cancer


df_sample.to_csv(join(PATH['CSV'], 's' + str(17) + '.csv'), index=False)
print(df_sample.head())

df_sample

df_sample = pd.read_csv(join(PATH['CSV'], 'stage2_sample_submission.csv'))   

df = pd.read_csv(join(PATH['STAGE_MASKS'], 'DATAFRAMES', 'stats_not_full.csv')).drop('Unnamed: 0', axis=1)


# df = add_modes(df.drop(['iso_modes'], axis=1), 
#                  df.iso_modes.values, 'iso_modes_')
# df = add_modes(df.drop(['vas_modes'], axis=1), 
#                  df.vas_modes.values, 'vas_modes_')
# df = add_modes(df.drop(['plu_modes'], axis=1), 
#                  df.plu_modes.values, 'plu_modes_')
# df['cancer'] = -1
# df = pd.concat([train, df])
df = df[df.cancer == -1]

s_list = sorted(clf.booster().get_fscore().items(),key=lambda x: x[1])
df = pd.DataFrame({'hui':[a[0] for a in s_list],'net':[a[1] for a in s_list]})
plt.figure()
df.plot()
df.plot(kind='barh', x='hui', y='net', legend=False, figsize=(10, 40))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.show()



