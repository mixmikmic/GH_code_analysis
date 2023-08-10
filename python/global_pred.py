import glob
import pandas as pd
import numpy as np
import json
import re

# Natural Sorting
# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside

def atoi(text):
    return int(text) if text.isdigit() else 0 #text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

folder = '../data/07-20-Data/gait'  #TODO: replace with the file you store gait json files
output = 'gait_config.csv'          #TODO: type in how you want to call the output file

fieldnames = ["title","offset","f1","f2","f3","f4","f5","t1","t2","t3","t4","t5"]
df = pd.DataFrame(columns = fieldnames)

gaitfiles = glob.glob(folder+'/*.json')
gaitfiles.sort(key=natural_keys)    # the csv files are named by date-time created
for gaitfile in gaitfiles:
    # Open json file
    with open(gaitfile) as json_file:    
        data = json.load(json_file)
    # Create a new row
    new_row = [str(data["Title"]), int(data["Offset"])]
    new_row += eval(data["Femur Sequence"])
    new_row += eval(data["Tibia Sequence"])
    # Merge the new row into dataframe
    df.loc[len(df)] = new_row

# Export dataframe into a csv file
df.to_csv(output,index=False)

