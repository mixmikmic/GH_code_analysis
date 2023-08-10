dictFeaIndex = {}
def preProcess(path):
    file = open(path + '.preProcess', 'w')
    for index, line in enumerate(open(path)):
        if index == 0:
            lstHead = line.strip('\n').split(',')            
            for idx, fea in enumerate(lstHead):
                dictFeaIndex[fea] = idx
            lstHead = lstHead + ['x_100', 'y_100', 'x_1000', 'y_1000', 'timeH']
            file.write(','.join(lstHead) + '\n')
            continue
        lst = map(float, line.strip('\n').split(','))
        x_100 = int(lst[dictFeaIndex['x']] * 100)
        y_100 = int(lst[dictFeaIndex['y']] * 100)
        x_1000 = int(lst[dictFeaIndex['x']] * 1000)
        y_1000 = int(lst[dictFeaIndex['y']] * 1000)
        timeH = int(lst[dictFeaIndex['time']] / 3600) % 24 / 24.000 * 5
        lst[dictFeaIndex['row_id']] = int(lst[dictFeaIndex['row_id']])
        lst = lst + [x_100, y_100, x_1000, y_1000, timeH]
        file.write(','.join(map(str, lst)) + '\n')
    file.close()
    
preProcess('data/train.csv')

import pandas as pd
print "!!!"
path = 'data/train.csv.preProcess'
dfAll = pd.read_csv(path)
uniqRowID = dfAll.row_id.unique()
trainThres = int(len(uniqRowID) * 0.7)
dfTrain = dfAll.loc[lambda df: df.row_id <= trainThres, :]
dfTest = dfAll.loc[lambda df: df.row_id > trainThres, :]
dfTrain.to_csv(path + '.train')
dfTest.to_csv(path + '.test')

import pandas as pd
from sklearn.neighbors import KDTree

def genCandidates(path, topK = 5):
    file = open(path + '.cand', 'w')
    dfAll = pd.read_csv('data/train.csv.preProcess.train')
    tree = KDTree(dfAll[['x', 'y','timeH']])
    dictFeaIndex = {}
    for index, line in enumerate(open(path)):
        if index == 0:
            lstHead = line.strip('\n').split(',')
            for idx, fea in enumerate(lstHead):
                dictFeaIndex[fea] = idx
            lstHead.append('place_id_cand')
            file.write(','.join(map(str, lstHead)) + '\n')
            continue
        lst = line.strip('\n').split(',')
        lstFea = [lst[dictFeaIndex['x']], lst[dictFeaIndex['y']], lst[dictFeaIndex['timeH']]]
        _, lstIndex = tree.query([lstFea], k = topK)
        for index in lstIndex:
            place_id = list(dfAll.iloc[index].place_id.values)[0]
            #print place_id
            if place_id != lst[dictFeaIndex['place_id']]:
                lst.append(int(place_id))
                file.write(','.join(map(str, lst)) + '\n')
    file.close()
print '###'    
genCandidates('data/train.csv.preProcess.train')



