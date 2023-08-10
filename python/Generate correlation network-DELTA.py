from IPython.core.display import display, HTML

display(HTML('''<style>
.container {width:98% !important;}
.dataframe th{font: bold 14px times; background: #0ea; text-align: right;}
.dataframe td{font: 14px courier; background: #fff; text-align: right;}
.output_subarea.output_text.output_stream.output_stderr {background: #fff; font-style: italic;}
</style>'''))

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

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
    log_path = 'correlation-delta.log'
    print "logging to %s" % log_path
    logging.basicConfig(filename=log_path,level=logging.DEBUG, format=FORMAT)
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

get_ipython().magic('matplotlib inline')
import pandas, pandas.io
import re
import seaborn as sns
import math
import scipy, scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import string
import os, os.path
logging.getLogger('boto').setLevel(logging.INFO)
logging.getLogger('p100').setLevel(logging.INFO)

import correlationsnodb.datasource as datasource

import correlationsnodb.analysis as analysis

DS_ID_MAP= '/home/jovyan/work/data/ds_id_map.pkl'                                                                                                                                                                                                             
PART_DF = '/home/jovyan/work/data/participant_data.pkl' 
DATA_DIR = '/home/jovyan/work/data'


import datetime
import itertools
def generate_correlations(args):
    rnd, entropy = args
    dsf = datasource.DataSourceFactory(ds_id_map=DS_ID_MAP, part_df=PART_DF, data_dir=DATA_DIR)
    logging.getLogger('p100.utils.correlations').setLevel(logging.DEBUG)
    # get all pairwise datasources
    bc = dsf.get_all_comparisons()
    analy = analysis.Analysis(ds_id_map=DS_ID_MAP, part_df=PART_DF, data_dir=DATA_DIR)
    #DEBUG
    test_ctr = 0
    for c1, c2 in bc:
        try:
            analy.Correlate( c1, c2, mean=False, mean_age_sex=False, delta_age_sex=True, tests=[analy.spearman], cutoff=1.01)
        except:
            logging.exception("Error correlating %s and %s for round %s" % ( str(c1), str(c2), rnd) )
            raise
        logging.debug("Completed comparisong %s and %s for round %s" % ( str(c1), str(c2), rnd))
    logging.info( "Correlations %s round(%s) entropy(%0.1f)" % (datetime.date.isoformat(datetime.datetime.now()),str(rnd), entropy) )
    return analy

analysis = generate_correlations(('delta', 0.75))

r = analysis.GetResult()
r.head()

r.shape

r = analysis.GetResult(annotated=True)
r.head()

r = analysis.GetResult(annotated=True, entropy=True)
r.head()



