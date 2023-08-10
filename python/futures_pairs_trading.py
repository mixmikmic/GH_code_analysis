from quantrocket.master import fetch_listings
fetch_listings(exchange="NYMEX", sec_types=["FUT"], symbols=["CL", "RB"])

from quantrocket.master import download_master_file, create_universe
import io

# Download the listings we just fetched
f = io.StringIO()
download_master_file(f, exchanges=["NYMEX"], sec_types=["FUT"], symbols=["CL", "RB"])
# then create a universe
create_universe("cl-rb", infilepath_or_buffer=f)

from quantrocket.history import create_db, fetch_history
create_db("cl-rb-1d", universes=["cl-rb"], bar_size="1 day")

fetch_history("cl-rb-1d")

from quantrocket.zipline import ingest_bundle
ingest_bundle(history_db="cl-rb-1d", calendar="us_futures")

from quantrocket.zipline import run_algorithm
import pandas as pd
import pyfolio as pf
import io

f = io.StringIO()
run_algorithm("futures_pairs_trading.py", 
              bundle="cl-rb-1d",
              start="2015-02-04", 
              end="2017-06-30",
              filepath_or_buffer=f)
results = pd.read_csv(f, parse_dates=["date"], index_col=["dataframe", "index", "date", "column"])["value"]

# Extract returns
returns = results.loc["returns"].unstack()
returns.index = returns.index.droplevel(0).tz_localize("UTC")
returns = returns["returns"].astype(float)

# Extract positions
positions = results.loc["positions"].unstack()
positions.index = positions.index.droplevel(0).tz_localize("UTC")
positions = positions.astype(float)

# Extract transactions
transactions = results.loc["transactions"].unstack()
transactions.index = transactions.index.droplevel(0).tz_localize("UTC")
transactions = transactions.apply(pd.to_numeric, errors='ignore')

# Extract benchmark returns
benchmark_returns = results.loc["benchmark"].unstack()
benchmark_returns.index = benchmark_returns.index.droplevel(0).tz_localize("UTC")
benchmark_returns = benchmark_returns["benchmark"].astype(float)

pf.create_full_tear_sheet(returns, positions=positions, transactions=transactions, benchmark_rets=benchmark_returns)

perf = results.loc["perf"].unstack()
perf.index = perf.index.droplevel(0).tz_localize("UTC")
perf = perf.apply(pd.to_numeric, errors='ignore')
print(perf.head())



