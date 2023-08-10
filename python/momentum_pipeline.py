from quantrocket.master import fetch_listings
symbols = ["SPY","EEM","GDX","VNQ","XLF","XOP"]
fetch_listings(exchange="ARCA", sec_types=["ETF"], symbols=symbols)

from quantrocket.master import download_master_file, create_universe
import io

# Download the listings we just fetched
f = io.StringIO()
download_master_file(f, exchanges=["ARCA"], sec_types=["ETF"], symbols=symbols)
# then create a universe
create_universe("etf-sampler", infilepath_or_buffer=f)

from quantrocket.history import create_db, fetch_history
create_db("etf-sampler-1d", universes=["etf-sampler"], bar_size="1 day")

fetch_history("etf-sampler-1d")

from quantrocket.zipline import ingest_bundle
ingest_bundle(history_db="etf-sampler-1d")

from quantrocket.zipline import run_algorithm
import pandas as pd
import pyfolio as pf
import io

f = io.StringIO()
run_algorithm("momentum_pipeline.py", 
              bundle="etf-sampler-1d",
              start="2015-02-04", 
              end="2015-12-31",
              filepath_or_buffer=f)

from quantrocket.zipline import ZiplineBacktestResult
zipline_results = ZiplineBacktestResult.from_csv(f)
print(zipline_results.returns.head())

pf.create_full_tear_sheet(
    zipline_results.returns, 
    positions=zipline_results.positions, 
    transactions=zipline_results.transactions, 
    benchmark_rets=zipline_results.benchmark_returns
)

print(zipline_results.perf.head())



