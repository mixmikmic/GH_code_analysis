import os
import glob
import pandas as pd
# contains population data.

os.chdir('C:\\Users\\577731\\Desktop\\va-datathon-2017\\data\\acsdata2')

acs_csv = [i for i in glob.glob('*.{}'.format('csv'))]
print(acs_csv)

trythis = acs_csv[0][4:6]
pd.read_csv(acs_csv[0],header=1)

def ingest_and_merge(directory,skiplines=2):
    os.chdir(directory)
    acs_csv = [i for i in glob.glob('*.{}'.format('csv'))]
    leftdf  = pd.read_csv(acs_csv[0], header=1)
    leftdf["year"] = '20'+acs_csv[0][4:6]
    rightdf = pd.read_csv(acs_csv[1], header=1)
    rightdf["year"] = '20'+acs_csv[1][4:6]
    growingchain = pd.merge(leftdf, rightdf, how='outer', on=['Id2','year'])
    for fileindex in range(2,len(acs_csv)):
        rightdf = pd.read_csv(acs_csv[fileindex], header=1)
        rightdf["year"] = '20'+acs_csv[fileindex][4:6]
        growingchain = pd.merge(growingchain, rightdf, how='outer', on=['Id2','year'])
    growingchain = growingchain.T.drop_duplicates().T
    return growingchain

masterdf = ingest_and_merge('C:\\Users\\577731\\Desktop\\va-datathon-2017\\data\\acsdata2')

def qualitycheck(df):
    columnlist = []
    nulllist = []
    columnlist = df.columns.tolist()
    for i in columnlist:
        print(i)
        nulls= sum(df[i].isnull())
        nulllist.append(nulls)
    df2 = pd.DataFrame({'column':columnlist, 'nulls': nulllist})
        
    return df2

masterdf.columns
#m2asterdf = masterdf

#m2asterdf = m2asterdf.T.drop_duplicates().T
#m2asterdf.shape



test = qualitycheck(masterdf)

test

masterdf.to_csv("ACS_master2.csv")

