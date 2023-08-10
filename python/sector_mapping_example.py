get_ipython().magic('matplotlib inline')
import pyfolio as pf
import gzip
import os
import pandas as pd

transactions = pd.read_csv(gzip.open('../tests/test_data/test_txn.csv.gz'),
                    index_col=0, parse_dates=0)
positions = pd.read_csv(gzip.open('../tests/test_data/test_pos.csv.gz'),
                    index_col=0, parse_dates=0)
returns = pd.read_csv(gzip.open('../tests/test_data/test_returns.csv.gz'),
                    index_col=0, parse_dates=0, header=None)[1]

returns.index = returns.index.tz_localize("UTC")
positions.index = positions.index.tz_localize("UTC")
transactions.index = transactions.index.tz_localize("UTC")

positions.head(2)

sect_map = {'COST': 'Consumer Goods', 
            'INTC': 'Technology', 
            'CERN': 'Healthcare', 
            'GPS': 'Technology',
            'MMM': 'Construction', 
            'DELL': 'Technology', 
            'AMD': 'Technology'}

pf.create_position_tear_sheet(returns, positions, sector_mappings=sect_map)

pf.create_round_trip_tear_sheet(positions, transactions, sector_mappings=sect_map)

