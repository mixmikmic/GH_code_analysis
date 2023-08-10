# Get backtest object
bt = get_backtest('536a6d181a4f090716c383b7')

# Create all tear sheets
bt.create_full_tear_sheet()

bt.create_returns_tear_sheet(live_start_date='2010-1-1')

bt.create_bayesian_tear_sheet(live_start_date='2011-04-01')

import pyfolio as pf

returns = bt.daily_performance.returns
pf.timeseries.cum_returns(returns).plot();

