import numpy
from collections import deque
import pandas
import math

def iterate_fund(ladder, yield_curve, max_maturity):
    reduce_maturity(ladder)
    
    payments = get_payments(ladder)

    sold_bond = ladder.popleft()
    payments += sold_bond.value(yield_curve)

    new_bond = Bond(payments, yield_curve[max_maturity-1], max_maturity)
    ladder.append(new_bond)
    
    # This happens *after* we sell the shortest bond and buy a new long one
    # (at least, that's what longinvest does...)
    nav = get_nav(ladder, yield_curve)

    return (ladder, payments, nav)

def get_nav(ladder, rates):
    return sum((b.value(rates) for b in ladder))

def get_payments(ladder):
    return sum((b.gen_payment() for b in ladder))

def reduce_maturity(ladder):
    for b in ladder:
        b.maturity -= 1
    return ladder

class Bond:
    def __init__(self, face_value, yield_pct, maturity):
        self.face_value = face_value
        self.yield_pct = yield_pct
        self.maturity = maturity
        
    def __repr__(self):
        return ('Maturity: %d | Yield: %.2f%% | Face Value: $%.2f' % (self.maturity, self.yield_pct * 100, self.face_value))
        
    def gen_payment(self):
        return self.face_value * self.yield_pct
    
    def value(self, rates):
        value = numpy.pv(rates[self.maturity - 1], self.maturity, self.gen_payment(), self.face_value)
        return -value

def bootstrap(yield_curve, max_bonds, min_maturity):
    bond_yield = yield_curve[max_bonds - 1]
    ladder = deque()
    starting_face_value = 50 # chosen arbitrarily (to match longinvest)

    for i, j in zip(range(max_bonds), range(min_maturity, max_bonds+1)):
        face_value = pow(1 + bond_yield, i) * starting_face_value
        b = Bond(face_value, bond_yield, j)
        ladder.append(b)
    return ladder
bootstrap([.0532]*10, 10, 2)

HISTORICAL_RATES = pandas.read_csv('bond_rates.csv', index_col=0)
HISTORICAL_RATES.head()

def splice_data(raw_rates, series):
    # Start by loading the data we get from Shiller.
    # This will always exist.
    series.iloc[0] = raw_rates['1 year']
    series.iloc[9] = raw_rates['10 year']
    
    # Try to load any FRED rates.
    series.iloc[1] = raw_rates['GS2']
    series.iloc[2] = raw_rates['GS3']
    series.iloc[4] = raw_rates['GS5']
    series.iloc[6] = raw_rates['GS7']
    series.iloc[19] = raw_rates['GS20']
    series.iloc[29] = raw_rates['GS30']
    
    def safe_add(series_index, rate_index):
        # Don't overwrite any data we already have.
        if math.isnan(series.iloc[series_index]):
            series.iloc[series_index] = raw_rates[rate_index]

    # These are in order of preference. This is try to use M13058 before
    # trying to use M1333.
    safe_add(19, 'M13058')
    safe_add(19, 'M1333')
    
    # See the note below under "Going Beyond 30 Years" about how longinvest got these numbers
    safe_add(19, 'longinvest 20')
    safe_add(29, 'longinvest 30')

def build_yield_curve(raw_rates, yield_curve_size=30):
    s = pandas.Series(math.nan, index=numpy.arange(yield_curve_size))

    # We use NaN to indicate "the data needs to be interpolated"
    # We have a few different data series that we splice together.
    splice_data(raw_rates, s)
    
    def left_number(series, index):
        """ Find the index of first number to the left """
        if not math.isnan(series.iloc[index]):
            return index
        else:
            return left_number(series, index-1)
        
    def right_number(series, index):
        """ Find the index of the first number to the right """
        if not math.isnan(series.iloc[index]):
            return index
        else:
            return right_number(series, index+1)
                
    # now fill in the gaps with linear interpolation.
    for i in range(yield_curve_size):
        if math.isnan(s.iloc[i]):
            # First, try to find any existing data on the left and right.
            # We might not find any, for instance when we look beyond 10-years
            # before we have FRED data.

            try:
                left = left_number(s, i)
            except IndexError:
                left = None
                
            try:
                right = right_number(s, i)
            except IndexError:
                right = None

            if (left is None) and (right is None):
                raise IndexError("Couldn't find any rate data to fill out the yield curve.")

            if left is None:
                # If we can't find any data to the left then we can't do any linear interpolation
                # So just fill from the right
                s.iloc[i] = s.iloc[right]
            elif right is None:
                # If we can't find any data to the right then fill from the left
                # Both of these will result in a flat yield curve, which isn't ideal
                s.iloc[i] = s.iloc[left]
            else:
                # We can actually do linear interpolation
                steps = right - left
                rate = s.iloc[left] + ((s.iloc[right] - s.iloc[left]) * (i - left) / steps)
                s.iloc[i] = rate

    return s.tolist()

['%.2f' % (s*100) for s in build_yield_curve(HISTORICAL_RATES.iloc[0])]

bootstrap(build_yield_curve(HISTORICAL_RATES.iloc[0]), 10, 4)

def loop(ladder, rates, max_maturity, start_year, end_year):
    df = pandas.DataFrame(columns=['NAV', 'Payments', 'Change'], index=numpy.arange(start_year, end_year + 1))

    for (year, current_rates) in rates:
        (ladder, payments, nav) = iterate_fund(ladder, build_yield_curve(current_rates), max_maturity)
        df.loc[year] = {'NAV' : nav, 'Payments' : payments}

    calculate_returns(df)
    return df

def calculate_returns(df):
    # Longinvest calculates the return based on comparison's to
    # next year's NAV. So I'll do the same. Even though that seems
    # weird to me. Maybe it's because the rates are based on January?
    # Hmmm...that sounds plausible.
    max_row = df.shape[0]

    for i in range(max_row - 1):
        next_nav = df.iloc[i+1]['NAV']
        nav = df.iloc[i]['NAV']
        change = (next_nav - nav) / nav
        df.iloc[i]['Change'] = change
    return df

def simulate(max_maturity, min_maturity, rates):
    """ This is just something to save on typing...and make clearer what the bounds on the fund are """
    ladder = bootstrap(build_yield_curve(rates.iloc[0]), max_maturity, min_maturity)
    start_year = int(rates.iloc[0].name)
    end_year = int(rates.iloc[-1].name)
    return loop(ladder, rates.iterrows(), max_maturity, start_year, end_year)

simulate(10, 2, HISTORICAL_RATES).head()

simulate(10, 4, HISTORICAL_RATES).head()

simulate(4, 2, HISTORICAL_RATES).head()

simulate(3, 2, HISTORICAL_RATES).head()

simulate(10, 1, HISTORICAL_RATES).head()

simulate(2, 2, HISTORICAL_RATES).head()

simulate(10, 10, HISTORICAL_RATES).head()

simulate(30, 11, HISTORICAL_RATES).head()

simulate(28, 14, HISTORICAL_RATES).head()

simulate(17, 9, HISTORICAL_RATES).head()

#simulate(10, 5, HISTORICAL_RATES).to_csv('10-5.csv')

import numpy
from collections import deque
import pandas
import math
import pandas_datareader.data as web
import datetime
import requests
import requests_cache
import xlrd
import tempfile

def get_morningstar(secid):
    url = 'http://mschart.morningstar.com/chartweb/defaultChart?type=getcc&secids=%s&dataid=8225&startdate=1900-01-01&enddate=2016-11-18&currency=&format=1' % secid
    expire_after = datetime.timedelta(days=3)
    session = requests_cache.CachedSession(cache_name='data-cache', backend='sqlite', expire_after=expire_after)

    # TODO: why doesn't this work!?!
    #r = session.get(url)
    r = requests.get(url)
    j = r.json()
    
    # The Morningstar data is pretty deeply nested....
    m = j['data']['r'][0]
    assert m['i'] == secid
    
    actual_data = m['t'][0]['d']
    # convert from strings to real data types
    as_dict = dict([(datetime.datetime.strptime(n['i'], '%Y-%m-%d'), float(n['v'])) for n in m['t'][0]['d']])
    
    # Strip out data?
    # Do we only want start of month, end of month, start of year, end of year, etc?
    s = pandas.Series(as_dict, name=secid)

    return s

barclays_index = get_morningstar('XIUSA000MJ')

# Use only final value for each calendar year
def annual(series):
    return series.groupby(by=lambda x: x.year).last()
# Use only final value for each calendar month
def monthly(series):
    return series.groupby(by=lambda x: datetime.date(x.year, x.month, 1)).last()

def calculate_change_prev(df, column):
    max_row = df.shape[0]
    
    series = pandas.Series()

    for i in range(max_row - 1):
        val = df.iloc[i][column]
        prev_val = df.iloc[i-1][column]
        change = (val - prev_val) / prev_val
        series.loc[df.iloc[i].name] = change
    return series

def calculate_change_next(df, column):
    max_row = df.shape[0]
    
    series = pandas.Series()

    for i in range(max_row - 1):
        val = df.iloc[i][column]
        next_val = df.iloc[i+1][column]
        change = (next_val - val) / val
        series.loc[df.iloc[i].name] = change
    return series

barclays_index = annual(barclays_index)

sim_10_4 = simulate(10, 4, HISTORICAL_RATES)
sim_10_10 = simulate(10, 10, HISTORICAL_RATES)

joined = pandas.concat([sim_10_4, barclays_index], axis=1, join='outer')
s_ind = calculate_change_prev(joined, 'XIUSA000MJ')
s_nav = calculate_change_next(joined, 'NAV')
joined = joined.assign(Change=s_nav, index_change=s_ind)
joined.to_csv('check.csv')
print(joined[["Change", "index_change"]].corr())

joined = pandas.concat([sim_10_10, barclays_index], axis=1, join='outer')
s_ind = calculate_change_prev(joined, 'XIUSA000MJ')
s_nav = calculate_change_next(joined, 'NAV')
joined = joined.assign(Change=s_nav, index_change=s_ind)
print(joined[["Change", "index_change"]].corr())



