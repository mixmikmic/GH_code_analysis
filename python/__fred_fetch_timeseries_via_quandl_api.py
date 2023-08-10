# TODOs:
    # load FRED-datasets-codes.csv, extract all codes
    # fetch all data by the codes, export to csv
    # export all time series data into HBase
    # implement in spark-ts and Hbase
    # dealing with spark-ts in Scala (optional)
    # https://blog.cloudera.com/blog/2015/12/spark-ts-a-new-library-for-analyzing-time-series-data-with-apache-spark/

import os
import time
import random
import quandl
import pandas as pd

file_path = "/Users/sundeepblue/Desktop/FRED-datasets-codes.csv"
save_path = "/Users/sundeepblue/Desktop/fred_codes/"
all_fred_codes = pd.read_csv(file_path, sep=',', header=None, names=["code_name", "description"])

def export_one_fred_code_to_csv(code_name, save_path):
    """
    code_name: the FRED code name, eg: 'FRED/DOTSRG3Q086SBEA'
    """
    parsed_code = "_".join(code_name.split("/"))
    file_name = "{}.csv".format(parsed_code)
    full_path = os.path.join(save_path, file_name)
    
    # call Quandl API to get time series
    try:  
        series = quandl.get(code_name)
    except Exception as e:
        print e
        series = None
    
    succeed = False
    # export series to csv file
    if series is not None:
        series.to_csv(full_path)
        succeed = True
        #print "\tExport to csv succeed!"
    else:
        print "\tFailed to fetch code {}".format(code_name)
    return succeed

def export_all_fred_codes(all_fred_codes, save_path):
    code_names = all_fred_codes.code_name
    failed_codes = []
    for i, c in enumerate(code_names):
        if (i+1) % 50 == 0:
            print i+1, c
        status = export_one_fred_code_to_csv(c, save_path)
        if status == False:
            failed_codes.append(c)
        time.sleep(random.uniform(0, 0.25))
    return failed_codes

quandl.ApiConfig.api_key = 'vzT_M2k5uyw3mz1z3ynR'

# Note: It took nearly 4 hours to fetch around 40000 time series (~250Mb) via quandl api on my macbook. 
# But there are in total 194824 FRED series. I manually interrupt the fetching because it seemed unnecessary
# for me to download all of them

export_all_fred_codes(all_fred_codes, save_path)

# let us now separately fetch the INDEX_GSPC-S-P-500-Index and export to csv
code = "YAHOO/INDEX_GSPC"
save_path = "/Users/sundeepblue/Desktop/gspc_code/"
export_one_fred_code_to_csv(code, save_path)

# Now we have necessary data to use in HBase

