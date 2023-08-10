# a CARD TXT parser
def TXTparser(filename):
    #filename = '1284787.3' ##
    f = open(filename+'.txt')
    listOfARO = []
    for lines in f:
        x = lines.split("\t")
        if len(x[10])>3:
            if ',' in x[10]:
                aroID = x[10].split(',')
                aroID = aroID[0]
                #print(aroID)
            else: aroID = x[10]
            num = int(aroID[4:])
            listOfARO.append(num)
    f.close()
    return(listOfARO)

    

import pandas as pd
df = pd.DataFrame()

IDList = []
import os
for file in os.listdir("."):
    if file.endswith(".txt"):
        IDList.append(file[:-4])
IDList.remove('listmp3')

i = 0
for ID in IDList:
    listOfARO = TXTparser(ID)
    df.loc[i, 'ID'] = ID
    #print(listOfARO)
    for ARO in listOfARO:
        
        df.loc[i, ARO] = 1
        
    i = i+1

df.fillna(value = 0, inplace = True)
df


df.to_csv('../visualizeARO.csv')

