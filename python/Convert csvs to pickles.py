try:
    run_once
except NameError:
    run_once = False
if not run_once:
    run_once = True
    
    import time
    import logging
    reload(logging)
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logpath = 'convert-csvs-to-pickle.log'
    logging.basicConfig(filename=logpath,level=logging.DEBUG, format=FORMAT)
    print("logging to %s" % (logpath))
    logger = logging.getLogger()
    #logger.basicConfig(filename='/notebooks/Export Microbiome to database.log',level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)


import os
import json
import pandas, pandas.io
print pandas.__version__

# the base of the data directory in the jupyter notebook
DATA_DIR = '/home/jovyan/work/data'
# the base of the csv directory in the jupyter notebook
CSV_DIR = '/home/jovyan/work/data/csv'

raise Exception("You probably don't want to run this")

def create_csv_and_json(sub, pickle_file):
    df = pandas.read_pickle(os.path.join(DATA_DIR, sub, pickle_file))
    # create a dict to capture important metadata
    meta = {'index_name':df.index.name}
    
    if isinstance(df, pandas.Series):
        meta['type'] = "Series"
        meta['dtype'] = str(df.dtypes)
    elif isinstance(df, pandas.DataFrame):
        meta['type'] = "DataFrame"
        meta['columns_name'] = df.columns.name
        meta['dtypes'] = {k:str(v) for k,v in df.dtypes.to_dict().items()}
        meta['index_dtype'] = str(df.index.dtype)
        meta['columns_dtype'] = str(df.columns.dtype)
    else:
        raise Exception("WTF")
    csv_name = pickle_file[:-3] + 'csv'
    json_name = pickle_file[:-3] + 'json'
    csv_path = os.path.join(CSV_DIR, sub, csv_name)
    json_path = os.path.join(CSV_DIR, sub, json_name)
    
    if isinstance(df.index, pandas.MultiIndex):
        raise Exception("WTF")
    df.to_csv(csv_path)
    print "Wrote", csv_path
    with open(json_path, 'w') as js:
        json.dump(meta, js)
    print "Wrote", json_path
        

        
# traverse root directory, and list directories as dirs and files as files
for root, dirs, files in os.walk(DATA_DIR):
    path = root.split(os.sep)
    sub = root[len(DATA_DIR):]
    #print((len(path) - 1) * '---', os.path.basename(root))
    if sub.find('csv') == -1:
        for fil in files:
            if fil[-3:] == 'pkl':

                #print(len(path) * '---', fil)
                #print sub
                create_csv_and_json(sub.lstrip(os.sep), fil)

def create_pickle(sub, csv_file):
    json_file = csv_file[:-3] + 'json'
    print csv_file
    meta = json.load(open(os.path.join(CSV_DIR, sub, json_file), 'r'))
    # print meta
    if meta['type'] == 'Series':
        series = pandas.read_csv(os.path.join(CSV_DIR, sub, csv_file), 
                        float_precision='high',
                        index_col = 0,
                        header=None
                       )
        if meta['index_name'] is not None:
            series.index.name = meta['index_name']
        series = series[series.columns[0]]
        series = series.astype(meta['dtype'])
        series.to_pickle(os.path.join(CSV_DIR, sub, csv_file[:-3] + 'pkl'))
        print "Wrote", os.path.join(CSV_DIR, sub, csv_file[:-3] + 'pkl')
        #print series.head()
    else:
        dataframe = pandas.read_csv(os.path.join(CSV_DIR, sub, csv_file), 
                        float_precision='high',
                        index_col = 0
                       )
        if meta['index_name'] is not None:
            dataframe.index.name = meta['index_name']
        if meta['columns_name'] is not None:
            dataframe.columns.name = meta['columns_name']
            
        dataframe.columns = dataframe.columns.astype(meta['columns_dtype'])

        dataframe = dataframe.astype({k:v for k,v in meta['dtypes'].items() if k in dataframe.columns})
        dataframe.index = dataframe.index.astype(meta['index_dtype'])
        if 'username' in dataframe.columns:
            dataframe = dataframe.astype({'username':str})
        dataframe.to_pickle(os.path.join(CSV_DIR, sub, csv_file[:-3] + 'pkl'))
        print "Wrote", os.path.join(CSV_DIR, sub, csv_file[:-3] + 'pkl')
        #print dataframe.head()
    
    
    
    
    
    

for root, dirs, files in os.walk(CSV_DIR):
    path = root.split(os.sep)
    sub = root[len(CSV_DIR):]
    print((len(path) - 1) * '---', os.path.basename(root))
    for fil in files:
        if fil[-3:] == 'csv':
            create_pickle(sub.lstrip(os.sep), fil)
            
            

import numpy as np
raise Exception("Are you sure? If so delete me")


for root, dirs, files in os.walk(DATA_DIR):
    path = root.split(os.sep)
    sub = root[len(DATA_DIR):]
    #print((len(path) - 1) * '---', os.path.basename(root))
    if sub.find('csv') == -1:
        for fil in files:
            if fil[-3:] == 'pkl':
                orig_df = pandas.read_pickle(os.path.join(root,fil))
                copied_df = pandas.read_pickle(os.path.join(CSV_DIR, sub.lstrip('/'), fil))
                if orig_df.equals(copied_df):
                    print os.path.join(root,fil), "==", os.path.join(CSV_DIR, sub.lstrip('/'), fil)
                else:
                    show_stopper = False
                    print os.path.join(root,fil), "!=", os.path.join(CSV_DIR, sub.lstrip('/'), fil)
                    if np.all(orig_df.dtypes == copied_df.dtypes):
                        print "Datatypes match"
                    else:
                        show_stopper = True
                    ocl = orig_df.columns.tolist()
                    ccl = copied_df.columns.tolist()
                    if len(ocl) == len(ccl) and ((set(ocl) & set(ccl)) == set(ccl)):
                        print "Columns match"
                    else:
                        show_stopper = True
                    ocl = orig_df.index.tolist()
                    ccl = copied_df.index.tolist()
                    if len(ocl) == len(ccl) and ((set(ocl) & set(ccl)) == set(ccl)):
                        print "Indices match"
                    else:
                        show_stopper = True
                    
                    nd = orig_df._get_numeric_data().columns.tolist()
                    nnd = [c for c in orig_df.columns if c not in nd]
                    if len(nnd) > 0:
                        if not np.all(orig_df[nnd] == copied_df[nnd]):
                            show_stopper = True
                            print "Nonumerics do not match"
                        else:
                            print "Nonnumerics match"
                    
                    if (copied_df._get_numeric_data() - orig_df._get_numeric_data()).abs().sum().sum() < .00001:
                        print "Numerics are Close enough"
                    else:
                        show_stopper = True
                    if show_stopper:
                        raise Exception("Out of range dataframes")



