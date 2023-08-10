import this

URL = "https://s3.amazonaws.com/pronto-data/open_data_year_one.zip"

import urllib.request
get_ipython().magic('pinfo urllib.request.urlretrieve')

import os
os.path.exists('open_data_year_one.zip')

# Python 2:
# from urllib import urlretrieve
# Python 3:
from urllib.request import urlretrieve
import os


def download_if_needed(url, filename, force_download=False):
    if force_download or not os.path.exists(filename):
        urlretrieve(url, filename)
    else:
        pass

    
download_if_needed(URL, 'open_data_year_one.zip')

get_ipython().system('ls')

from pronto_utils import download_if_needed
download_if_needed(URL, 'open_data_year_one.zip')

import zipfile
import pandas as pd

def load_trip_data(filename='open_data_year_one.zip'):
    """Load trip data from the zipfile; return as DataFrame"""
    download_if_needed(URL, filename)
    zf = zipfile.ZipFile(filename)
    return pd.read_csv(zf.open('2015_trip_data.csv'))

data = load_trip_data()
data.head()

from pronto_utils import load_trip_data
data = load_trip_data()
data.head()

import pandas as pd
from pronto_utils import load_trip_data

def test_trip_data():
    df = load_trip_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (142846, 12)
    
test_trip_data()

get_ipython().system('py.test pronto_utils.py')

get_ipython().magic('matplotlib inline')

def plot_totals_by_birthyear():
    df = load_trip_data()
    totals_by_birthyear = df.birthyear.value_counts().sort_index()
    return totals_by_birthyear.plot(linestyle='steps')

plot_totals_by_birthyear()

def test_plot_totals():
    ax = plot_totals_by_birthyear()
    assert len(ax.lines) == 1
    

import numpy as np
import matplotlib as mpl

def test_plot_totals_by_birthyear():
    ax = plot_totals_by_birthyear()
    
    # Some tests of the output that dig into the
    # matplotlib internals
    assert len(ax.lines) == 1
    
    line = ax.lines[0]
    x, y = line.get_data()
    assert np.all((x > 1935) & (x < 2000))
    assert y.mean() == 1456

test_plot_totals_by_birthyear()

get_ipython().system('py.test pronto_utils.py')



