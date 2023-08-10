# Get backtest object
bt = get_backtest('55db2def35e3b00d9986aa99')

# Create all tear sheets
bt.create_full_tear_sheet()

bt.create_returns_tear_sheet(live_start_date='2014-1-1')

bt.create_bayesian_tear_sheet(live_start_date='2014-1-1')

import pyfolio as pf

returns = bt.daily_performance.returns
pf.timeseries.cum_returns(returns).plot();

