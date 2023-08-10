import pandas
import portfolio
import collections
from decimal import Decimal
from adt import AnnualChange

Date = collections.namedtuple('Date', ['year', 'month'])

def str_to_date(s):
    (year, month) = s.split('.')
    def convert_month(m):
        if m == '01':
            return 1
        elif m == '1':
            return 10
        else:
            return int(m)
    return Date(int(year), convert_month(month))

frame = pandas.read_csv('shiller_monthly.csv', converters={'Date': str_to_date})

CAPE_STARTS = Date(1881, 1)
DATA_STARTS = Date(1871, 1)
MONTHS_IN_YEAR = 12
YEARS_IN_RETIREMENT = 30
LENGTH_OF_RETIRMENT = YEARS_IN_RETIREMENT * MONTHS_IN_YEAR

def get_row(date):
    years = date.year - DATA_STARTS.year
    months = years * MONTHS_IN_YEAR
    months += date.month - DATA_STARTS.month
    return months

assert frame.iloc[get_row(CAPE_STARTS)]['Date'] == CAPE_STARTS

def retire(frame, start_date):
    assert start_date.year >= CAPE_STARTS.year
    
    current_date = start_date
    
    p = portfolio.Portfolio(500000, 500000)
    last_equity_index = frame.iloc[get_row(start_date)]['S&P Price']

    for i in range(LENGTH_OF_RETIRMENT):
        df = frame.iloc[get_row(current_date)]
        
        # update portfolio
        current_equity_index = frame.iloc[get_row(current_date)]['S&P Price']
        percent_change = (current_equity_index / last_equity_index) - 1
        last_equity_index = current_equity_index
        
        dividends = frame.iloc[get_row(current_date)]['S&P Dividend']
        monthly_yield = (dividends / current_equity_index) / 12
        
        dollar_change = (p.stocks * Decimal(percent_change)) + (p.stocks * Decimal(monthly_yield))

        stock_change = Decimal(percent_change) + Decimal(monthly_yield)

        p.adjust_returns(AnnualChange(year=0, stocks=stock_change, bonds=0, inflation=0))

        if current_date.month == start_date.month:
            print(df['Date'], p.value)
            # make withdrawal based on CAPE
            # rebalance?
            
        new_date_months = current_date.year * MONTHS_IN_YEAR + current_date.month
        current_date = Date(new_date_months // 12, (new_date_months % 12) + 1)

#retire(frame, Date(1900, 1))

def make_csv(frame):
    pd = pandas.DataFrame(columns=['CAPE10', 'Mean', 'Median'])
    for row, i in frame.iloc[0:].iterrows():
        if i['Date'].month == 12 and i['Date'].year >= 1881:
            pd.loc[i['Date'].year] = {'CAPE10': i['CAPE10'], 'Mean': i['Mean'], 'Median': i['Median']}
    #pd.to_csv('cape10.csv')
    print(pd.head())
make_csv(frame)



