# Statistics
import pandas as pd

dfTrain = pd.read_csv('data/train.csv')
dfTest = pd.read_csv('data/test.csv')

xMin = dfTrain.x.min()
xMax = dfTrain.x.max()
yMin = dfTrain.y.min()
yMax = dfTrain.y.max()
print "xMin: %f, xMax: %f, yMin: %f, yMax: %f" % (xMin, xMax, yMin, xMax)

print dfTrain.shape

from sklearn.neighbors import KDTree
dfAll = pd.read_csv('data/train.csv.100000')

def preProcess(df):
    df['x_1000'] = df.apply(lambda x : int(x['x'] * 1000), axis=1) 
    df['x_100'] = df.apply(lambda x : int(x['x'] * 100), axis=1)
    df['y_1000'] = df.apply(lambda x : int(x['y'] * 1000), axis=1) 
    df['y_100'] = df.apply(lambda x : int(x['y'] * 100), axis=1)
    df['timeH'] = df.apply(lambda x: int(x['time'] / 3600) % 24 / 24.000 * 5, axis = 1)
    
def genCandidates(df, topK = 5):
    print 'genCandidates'
    
    tree = KDTree(dfAll[['x', 'y','timeH']])
    _, ind = tree.query(df[['x','y','timeH']], k = topK)
    lstDF = []
    for i in range(topK):
        dfTmp = df.copy()
        ind1 = [x[i] for x in ind]
        #print ind1
        dfTmp['place_id_cand'] = dfAll.iloc[ind1].place_id.values
        lstDF.append(dfTmp)
    df['place_id_cand'] = df['place_id']
    lstDF.append(df)
    df = pd.concat(lstDF, ignore_index = True)
    print len(df.index)
    #df['label'] = df['place_id'] == df['place_id_cand'] ? 1 : 0
    df['label'] = df.apply(lambda x : 1 if x['place_id'] == x['place_id_cand'] else 0, axis=1)  
    #df[['row_id', 'place_id']].to_csv('submission_time.gz', index = False, compression = 'gzip')
    return df

def genFeatures(df, lstTopK = [5]):
    # Depreciated, will not use KNN statis as features since too time consuming
    key = ['x', 'y']
    treeXY = KDTree(dfAll[key])
    for topK in lstTopK:
        print 'top%d_xy' % (topK)
        _, lstTopKIndex = treeXY.query(df[key], k = topK)
        df['top%d_xy' % (topK)] = [[dfAll.iloc[xx].place_id for xx in x].count(df.iloc[index].place_id_cand) for index, x in enumerate(lstTopKIndex)]
    key = ['x', 'y', 'timeH']
    treeXYTimeH = KDTree(dfAll[key])  
    for topK in lstTopK:
        print 'top%d_xyTimeH' % (topK)
        _, lstTopKIndex = treeXYTimeH.query(df[key], k = topK)
        df['top%d_xyTimeH' % (topK)] = [[dfAll.iloc[xx].place_id for xx in x].count(df.iloc[index].place_id_cand) for index, x in enumerate(lstTopKIndex)]
    
    print [dfAll.iloc[xx].place_id for xx in x]
    print df.iloc[index].place_id_cand


dictFeaAggKey = {
    'xy1000_placeId': ['x_1000', 'y_1000', 'place_id_cand'],
    'xy100_placeId': ['x_100', 'y_100', 'place_id_cand'],
    'xy1000_placeId_timeH' : ['x_1000', 'y_1000', 'place_id_cand', 'timeH'],
    'xy100_placeId_timeH' : ['x_100', 'y_100', 'place_id_cand', 'timeH'],
}

dictFeaAggDF = {}

def genFeaAgg(df):
    # The KDTree based feature is too time consuming
    df['place_id_cand'] = df['place_id']
    for fea in dictFeaAggKey.keys():
        keyAgg = dictFeaAggKey[fea]
        dfGroup = df.groupby(keyAgg).count().reset_index()
        dfGroup[fea] = dfGroup['x']
        dfGroup = dfGroup[keyAgg + [fea]]
        dictFeaAggDF[fea] = dfGroup

def MergeAggFea(df):
    dfMerge = df.copy()
    for fea in dictFeaAggDF.keys():
        dfAgg = dictFeaAggDF[fea]
        dfMerge = pd.merge(dfMerge, dfAgg, how = 'left', on = dictFeaAggKey[fea])
    return dfMerge

import random, time
dfAll = pd.read_csv('data/train.csv')
uniqRowID = dfAll.row_id.unique()
print 'preProcess'
timeStart = time.time()
preProcess(dfAll)
timeCurrent = time.time()
trainThres = int(len(uniqRowID) * 0.7)
dfTrain = dfAll.loc[lambda df: df.row_id <= trainThres, :]
dfTest = dfAll.loc[lambda df: df.row_id > trainThres, :]
print 'dfTrainCan'
dfTrainCan = genCandidates(dfTrain)
dfTrainCan = dfTrainCan.drop_duplicates()
print 'dfTestCan'
dfTestCan = genCandidates(dfTest)
dfTestCan = dfTestCan.drop_duplicates()
print 'genFeaAgg'
genFeaAgg(dfTrain)
print 'dfTrainAggFea'
dfTrainAggFea = MergeAggFea(dfTrainCan)
print 'dfTrainAggFea'
dfTestAggFea = MergeAggFea(dfTestCan)
timeCurrent = time.time()
print "%f s has passed" % (timeCurrent - timeStart)

from sklearn.ensemble import RandomForestClassifier
dfTrainAggFea = dfTrainAggFea.fillna(0)
clf = RandomForestClassifier(n_estimators=100, min_weight_fraction_leaf=0.1)
predictors = ['x', 'y', 'accuracy', 'time', 'xy100_placeId', 'xy1000_placeId', 'xy1000_placeId_timeH', 'xy100_placeId_timeH']
#predictors = ['x', 'y', 'accuracy', 'time', 'xy100_placeId']
clf.fit(dfTrainAggFea[predictors], dfTrainAggFea['label'])

dfTestAggFea = dfTestAggFea.fillna(0)
lstScore = clf.predict_proba(dfTestAggFea[predictors])
print lstScore[:10]

def genTopResult(df, lstScore, topK = 3):
    lstRes = []
    idx = -1
    dictRowIdRes = {}
    for _, row in df.iterrows():
        idx += 1
        rowId = row['row_id']
        placIdCan = row['place_id_cand']
        score = lstScore[idx][1]
        dictRowIdRes.setdefault(rowId, [])
        dictRowIdRes[rowId].append([placIdCan, score])
    rowIdTestUnique = sorted(df.row_id.unique())
    for rowId in rowIdTestUnique:
        #print 'dictRowIdRes[rowId]', dictRowIdRes[rowId]
        res = sorted(dictRowIdRes[rowId], key = lambda x: x[1], reverse = True)
        #print 'res', res
        lstRes.append([rowId, [x[0] for x in res[:topK]], res])
    return lstRes

def genLabel(df):
    lstLabel = []
    rowIdTestUnique = sorted(df.row_id.unique())
    for rowId in rowIdTestUnique:
        label = dfTest[dfTest.row_id == rowId].place_id.unique()[0]
        lstLabel.append([label])
    return lstLabel

lstRes0 = genTopResult(dfTestAggFea, lstScore)
lstRes = [map(int, x[1]) for x in lstRes0]
lstLabel = genLabel(dfTestAggFea)

import ml_metrics as metrics
metrics.mapk(lstLabel, lstRes, k = 3)

import numpy as np
importances = clf.feature_importances_
print len(importances)
print len(predictors)


indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(len(predictors)):
    print("%d. %str: %f" % (f + 1, predictors[f], importances[f]))



