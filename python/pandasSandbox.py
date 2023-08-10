import pandas as pd
import numpy as np
import os

import random
import string
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

os.chdir('/Users/Seth/Documents/DSI/Capstone/DSI-Religion-2017/')

rawPath = './data_dsicap/'
groups = os.listdir(rawPath)

print(groups)

naw = ['.DS_Store', 'test_train', 'paw', 'norun']
[groups.remove(x) for x in groups if x in naw]

groups

def createFileListRandom(groups, bin=5):
    import math
    rawFileList=[]
    for groupId in groups:
        for dirpath, dirnames, filenames in os.walk(rawPath+groupId+'/raw'):
            filecount = len([filename for filename in filenames if ".txt" in filename]) # how many files are there with .txt
            bincount = math.ceil(filecount/float(bin))
            subgroups = [groupId + '_test' + str(num) for num in range(0,int(bincount))]
            for filename in [f for f in filenames ]:
                if '.txt' in filename:
                    rawFileList.append([groupId,os.path.join(dirpath, filename), np.random.choice(subgroups)])
    return rawFileList

rawFileListRandom = createFileListRandom(groups)

dfR = pd.DataFrame(rawFileListRandom, columns=["group","filepath", "subgroups"])
print(dfR)

dfR.sort_values(by='subgroups')

def createFileList(groups, bin=5):
    import math
    from itertools import repeat
    rawFileList=[]
    for groupId in groups:
        for dirpath, dirnames, filenames in os.walk(rawPath+groupId+'/raw'):
            ## clean out non .txt files
            filenames = [filename for filename in filenames if ".txt" in filename]
            ## create subgroups
            filecount = len(filenames) # how many files are there in this groupId
            bincount = math.ceil(filecount/float(bin)) # how many bins do we need
            subgroups = [groupId + '_test' + str(num) for num in range(0,int(bincount))] # create bins
            sglist = [x for item in subgroups for x in repeat(item, bin)] # make list of bins (copy each option 5x (or whatever you put in the bin=))
            sglist = sglist[:filecount] # trim to the number of files you have
            random.shuffle(sglist) # shuffle them before assigning (CHANGE THIS IF WE MAKE IT TEMPORAL)
            ## append to rawFileList
            #for filename in filenames:
            #    rawFileList.append([groupId,os.path.join(dirpath, filename), 't'])
            for i in range(0,filecount):
                rawFileList.append([groupId,os.path.join(dirpath, filenames[i]), sglist[i]])

    #print(rawFileList)
    return rawFileList

rawFileList = createFileList(groups)

df = pd.DataFrame(rawFileList, columns=["group","filepath", "subgroups"])
print(df)

print(df.sort_values(by='subgroups'))

rawFileList=[]
for groupId in groups:
    for dirpath, dirnames, filenames in os.walk(rawPath+groupId+'/raw'):
        for filename in [f for f in filenames ]:
            if '.txt' in filename:
                rawFileList.append([groupId,os.path.join(dirpath, filename),'test-'+id_generator()])

print(rawFileList)

len(rawFileList)
print(type(rawFileList))

rfdf = pd.DataFrame(rawFileList, columns=["group","filepath","subgroup"])
print(rfdf)









data1 = {'group':['test1','train6','test2','train7','test3'],
        #'group2':['test10','train5','test6','train11','test7'],
        'svd':np.random.uniform(0,10,5).tolist(),
        'evc':np.random.uniform(0,10,5).tolist()}
data1

data2 = {'group':['train8','test4','train9','test5', 'train10'],
        'svd':np.random.uniform(0,10,5).tolist(),
        'evc':np.random.uniform(0,10,5).tolist()}
data2

df1 = pd.DataFrame(data1, columns=['group','svd','evc'])
df1

df2 = pd.DataFrame(data2, columns=['group','svd','evc'])
df2

df1.replace('test','train1', regex=True, inplace=True)
df2.replace('train','test1', regex=True, inplace=True)

df1

df2

df = df1.append(df2)
df

df[df['group'].str.contains("train")]

df.shape

df

mo = pd.read_csv('masterOutput.csv', index_col=0)
mo

groups = groups[0:3]
print(groups)


g2 = [x+'two' for x in groups]
stats = pd.DataFrame([tuple(groups),tuple(g2)], columns = ['one', 'two', 'three'])
print(stats)

stats.append(pd.DataFrame([tuple(groups),tuple(g2)], columns = ['one', 'two', 'four']))

stats.to_csv('modelOutput/statsTest.csv', index=False)





newStats = [x+'new' for x in groups] #######
try:
    modelStats = pd.read_csv('modelOutput/modelStats.csv')
    modelStats = modelStats.append(pd.DataFrame([tuple(newStats)], columns = ['one', 'two', 'three']))
    print("added row to modelStats.csv")
except:
    modelStats = pd.DataFrame([tuple(newStats)], columns = ['one', 'two', 'three'])
    print("created modelStats.csv file")
#
modelStats.to_csv('modelOutput/modelStats.csv', index=False)

stats.shape[0] + modelStats.shape[0]

naw = ['paw', 'haw']
jaw = ['aww','naw', 'yall']
law = naw + jaw
print(law)

yall = u'ya\u201dll'
print(yall)
yall = yall.replace(u'\u201d','')
print(yall)





posFilePath='./refData/positive-words.txt'
pos = list(set(unicode(open(posFilePath).read(), "utf-8", errors="ignore")))
print(pos)

posRaw = unicode(open(posFilePath).read(), "utf-8", errors="ignore")
posRaw

def split_line(text):

    # split the text
    words = text.split()

    # for each word in the line:
    for word in words:

        # print the word
        print(word)

split_line(posRaw)

posList = posRaw.split()

'acclaim' in posList

'acc' in posList



runDirectory = './modelOutput/'
signalFile = 'signalOutput-coco_3_cv_3_netAng_30_twc_20_tfidf_pronounFrac_bin_3-D25MTC.csv'
paramPath = signalFile.split('-')[1]
print(paramPath)

signalDF = pd.read_csv(runDirectory + 'logs/' + signalFile, index_col=0)
signalDF.head()

allCols = signalDF.columns.tolist()
print(allCols)

nawCols = ['groupId', 'files', 'timeRun', 'keywords','rank']
xList = [x for x in allCols if x not in nawCols]
print(xList)





