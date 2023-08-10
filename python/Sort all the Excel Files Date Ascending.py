import pandas as pd
import csv
import os
from os import listdir
from os.path import isfile, join

import settings_skyze

only_files = [f for f in listdir(settings_skyze.data_file_path) if isfile(join(settings_skyze.data_file_path, f))]
only_files.pop(0)
print(only_files)



count = 1
for market in only_files:
    
    # read in the file
    file_path = os.path.join(settings_skyze.data_file_path,  market)
    print(str(count)+". "+market)
    fileDF = pd.read_csv(file_path,header=None )
    # print(fileDF)
    # convert date column to datetime type
    fileDF[0] =  pd.to_datetime(fileDF[0], format='%b %d %Y')

    # sort it
    sortedDF = fileDF.sort([0], ascending=True)
    #print(sorted)

    # convert date column to string
    #sortedDF[0] =  pd.to_datetime(sortedDF[0], format='%b %d %Y')
    sortedDF[0] =  sortedDF[0].dt.strftime('%Y-%m-%d')
    #sortedDF[0].apply(lambda x: x.strftime('%b %d %Y'))

    # save it
    sortedDF.to_csv(file_path, index=False, header=False)
    count+=1



