import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd

import graphviz
import os
import time

np.__version__, pd.__version__, dask.__version__

# Support multiple lines of output in each cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Don't wrap tables
pd.options.display.max_rows = 20
pd.options.display.width = 200

# Show matplotlib graphs inline in Jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# The sample data is the USA Bureau of Transportation Statistics 'On-Time' monthly series.
# This has actual arrival/departure times versus schedule for every domestic flight
# by major US carriers. For details, see the BTS website:
#    https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236&DB_Short_Name=On-Time

# We can download the monthly results directly from this URL, filling in the
# two parameters with the year (e.g. '2016') and month ('1' for January, '12' for December).
# The resulting zip files, each around 23MB in size when compressed,
# contain a 200MB .csv file with the same name (On_Time_On_Time_Performance_2016_1.csv)
# plus a 'readme.html' explaining the fields.
OTP_URL = 'https://transtats.bts.gov/PREZIP/On_Time_On_Time_Performance_%s_%s.zip'

OTP_COLUMNS_TO_LOAD = [
        'FlightDate', 'Origin', 'Dest', 'Distance',
        'Carrier', 'FlightNum', 'TailNum',
        'CRSDepTime', 'CRSArrTime', 'CRSElapsedTime',
        'Flights', 'Cancelled','Diverted',
        'DepTime', 'ArrTime', 'ActualElapsedTime',
        'DepDelay', 'ArrDelay', 'AirTime',
    ]

# Directory to store the resulting .zip files
if os.path.exists('/home/stephen/do-not-backup'):
    DIR_NAME = '/home/stephen/do-not-backup/data/usa-flights-otp'
else:
    DIR_NAME = '~/pydata-pandas/data'

# Download USA flight data as described at 
# https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236&DB_Short_Name=On-Time
import logging
import multiprocessing
import ssl
import urllib.request

BASE_URL = 'https://transtats.bts.gov//PREZIP/On_Time_On_Time_Performance_%s_%s.zip'
DIR_NAME = '/home/stephen/do-not-backup/data/usa-flights-otp'

def retrieve_data(year_month):
    """
    Retrieve BTS on-time flight data for given year and month,
    unpack csv data from zip file and save as 'flights-yyyy-mm.xz'.
     
    year_month  - Month of data to retrieve, in form of a tuple of ints 
                    like (2016, 1) for January 2016.
    """
    os.makedirs(DIR_NAME, exist_ok=True)
    filename = 'flights-%04d-%02d' % year_month
    zip_path = os.path.join(DIR_NAME, filename + '.zip')
    xz_path  = os.path.join(DIR_NAME, filename + '.xz' )
    csv_filename = 'On_Time_On_Time_Performance_%s_%s.csv' % year_month

    if os.path.exists(xz_path):
        print("%s - Already have .xz file" % filename)
    else:
        started = time.time()
        # Get zip file's data
        if os.path.exists(zip_path):
            # Extract from previously downloaded zip file
            print("%s - Reading csv from %s" % (filename, zip_path))
            zip_src = zip_path
        else:
            # Download zip file to memory
            url = OTP_URL % year_month
            print("%s - Downloading %s" % (filename, url))
            # We would like to do simply this:
            #   urllib.request.urlretrieve(url, dest_path)
            # but that gives SSL errors
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            zip_data = urllib.request.urlopen(url, context=ctx).read() # 25MB
            zip_src = io.BytesIO(zip_data)
        # Extract csv data and recompress to .xz archive
        csv_data = zipfile.ZipFile(zip_src).open(csv_filename).read()  # 200MB!
        MB = 1024.0 * 1024.0
        print("%s - csv data is %0.1fMB. Now compressing..."
                        % (filename, len(csv_data) / MB))
        with lzma.open(xz_path, 'wb') as xzf:
            xzf.write(csv_data)
            csv_MB = xzf.tell() / MB
            xz_MB = xzf._fp.tell() / MB
        mins, secs = divmod(time.time() - started, 60)
        print("%s - Compressed csv from %0.1fMB to %0.1fMB [%02d:%02d, %0.1fMB mem]"
                        % (filename, csv_MB, xz_MB, mins, secs, memory_usage() ))


def download_flight_data(start='1988-01', end=None, num_threads=4):
    """
    Download BTS On-Time flight data for one month or a range of months.
    Data is available from '1987-12' to '2017-01' inclusive.
    The resulting zip files are named 'files-yyyy-mm.zip'.
    """
    end   = tuple(map(int, min(end or start, '2017-01').split('-')))
    start = tuple(map(int, max(start,        '1987-12').split('-')))

    dates = (
        (year, month)
            for year in range(end[0], start[0] - 1, -1)
                for month in range(12, 0, -1)
                    if start <= (year, month) <= end
    )

    multiprocessing.Pool(num_threads).map(retrieve_data, dates)

download_flight_data('2016-01', '2016-12')

path = os.path.join(DIR_NAME, 'flights-2017-01.xz')
df = pd.read_csv(path, dialect="excel", nrows=10)
df.info()

def memory_usage(log=False):
    """Return current memory usage or print in log. Requires `psutil` package installed."""
    pid = os.getpid()
    try:
        import psutil
        mem_MB = psutil.Process(pid).memory_info().rss / 1024.0 / 1024.0
        msg = "Memory for process %s is %0.1fMB" % (pid, mem_MB)
    except:
        mem_MB = None
        msg = "Process is pid %s. Memory usage unavailable" % pid
    if log:
        logging.info(msg)
    return mem_MB

def load_one_month(yyyy_mm, nrows=None):
    """
    Load one month's data as a pandas DataFrame. 
    Optionally limit max number of rows read.
    """
    started = time.time()

    # Load the csv from xz-compressed file
    path = os.path.join(DIR_NAME, 'flights-%s.xz' % yyyy_mm)
    df = pd.read_csv(path,
                     dialect="excel",
                     usecols=OTP_COLUMNS_TO_LOAD,
                     nrows=nrows,
                     parse_dates=['FlightDate'],
                     dtype={ 'FlightNum': str, }, # Keep as string, to later combine with carrier
                     )

    # Put columns in our standard order
    df = df[OTP_COLUMNS_TO_LOAD]
    df['FlightNum'] = df['Carrier'] + df['FlightNum']   # to give 'AA494'

    mm, ss = divmod(time.time() - started, 60)
    logging.info("Loading pd.DataFrame for %s took %02d:%02d (%dMB mem)", yyyy_mm, mm, ss, memory_usage())
    return df

get_ipython().run_line_magic('timeit', '')
df = load_one_month('2017-01')
df.info()

for col in ['FlightDate','Origin','Dest','Carrier','FlightNum','TailNum']:
    df[col] = df[col].astype('category', ordered=True)
df.info()

df['Carrier'].cat.categories

df

df.memory_usage(deep=True).sum()

