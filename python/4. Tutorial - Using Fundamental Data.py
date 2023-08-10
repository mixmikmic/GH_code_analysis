# First, we need to initialize fundamentals in any notebook that uses it

fundamentals = init_fundamentals()

# We're going to need a date string for some of our queries. Let's set one up

import datetime

today = datetime.datetime.now().strftime('%Y-%m-%d')
today

# The key function, like the backtester, is called get_fundamentals()
# Before we get started, you can access the docstring by executing this cell
get_ipython().magic('pinfo get_fundamentals')

# Let's do a simple query.

fund_df = get_fundamentals(query(fundamentals.valuation_ratios.pe_ratio)
                             .filter(fundamentals.valuation.market_cap > 1e9)
                             .filter(fundamentals.valuation_ratios.pe_ratio > 5)
                             .order_by(fundamentals.valuation.market_cap)
                             .limit(10),
                             today)

# OK, let's check out what we get back.
# When we provide a query and a date, we get back the same type of response
# as in the IDE: a dataframe with securities as columns and each requested
# metric as rows.

fund_df

# We can use pandas to examine and manipulate the data

fund_df.loc['pe_ratio']

# Let's try something more interesting.
# Let's ask for more than 1 metric and ask for a time series worth of data, like the last 100 days

fund_panel = get_fundamentals(query(fundamentals.valuation_ratios.pe_ratio,
                                   fundamentals.valuation.market_cap)
                             .filter(fundamentals.valuation.market_cap > 1e9)
                             .filter(fundamentals.valuation_ratios.pe_ratio > 5)
                             .order_by(fundamentals.valuation.market_cap.desc())
                             .limit(100),
                             today, '100d')
fund_panel

# When we ask for more than a single day's worth of data, we get a panel instead of a dataframe.
# Each dataframe within the panel contains data for a single metric included in the query.
# Rows in the dataframes represent the different dates. Columns represent different securities.
#
# Here, we are accessing just the PE Ratio metric. The columns
# are the individual securities (we've limited it to 100)
# The rows are different dates. We've got 100 different points
# in a time series because we asked for '100d', i.e 100 days worth of
# data points, starting from today and going backwards.


fund_panel.loc['pe_ratio']

# Another way to access this same dataframe of pe ratio data

fund_panel.iloc[0]

# Let's also check out the market cap

fund_panel.loc['market_cap']
fund_panel.iloc[1]

# Get a list of sids for the date upon which you want to enforce your filtering and limiting criteria

criteria_df = get_fundamentals(query(fundamentals.company_reference.sid
        )
        .filter(fundamentals.valuation.market_cap> 100000000)
        .order_by(fundamentals.valuation.market_cap.desc())
        .limit(10)
        , '2015-05-01')
series_of_securities = criteria_df.loc['sid']

series_of_securities

# Turn it into a list
securities = list(criteria_df.columns.values)

# Turn that list into a list of just the sids (as opposed to the full security object)

starting_sids = [stock.sid for stock in securities]
starting_sids

# Use those sids to get the data you want
criteria_panel = get_fundamentals(query(fundamentals.company_reference.sid,
                                        fundamentals.valuation.market_cap
        )
        .filter(fundamentals.company_reference.sid.in_((starting_sids))),
        '2015-05-01', '6m')

criteria_panel

criteria_panel.loc['market_cap']

