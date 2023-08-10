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

BOND_RATES = pandas.read_csv('oecd_bond_rates.csv', index_col=0)
BOND_RATES.head()

BILL_RATES = pandas.read_csv('oecd_bill_rates.csv', index_col=0)
BILL_RATES.head()

def build_yield_curve(bill_rate, bond_rate, yield_curve_size=10):
    s = pandas.Series(math.nan, index=numpy.arange(yield_curve_size))
    s.iloc[0] = bill_rate
    s.iloc[9] = bond_rate
    s.interpolate(inplace=True)
    s.fillna(method='backfill', inplace=True)    

    return s.tolist()

def get_rate_pair_at(year, country):
    bond_rate = BOND_RATES.loc[year][country]
    bill_rate = BILL_RATES.loc[year][country]
    return (bill_rate, bond_rate)

['%.2f' % (s*100) for s in build_yield_curve(*get_rate_pair_at(1970, 'AUS'))]

yield_curve = build_yield_curve(*get_rate_pair_at(1970, 'AUS'))
bootstrap(yield_curve, 10, 4)

def loop(ladder, rates, max_maturity, start_year, end_year):
    df = pandas.DataFrame(columns=['NAV', 'Payments', 'Change'], index=numpy.arange(start_year, end_year + 1))
    
    for year in range(start_year, end_year+1):
        c = rates.loc[year]
        (ladder, payments, nav) = iterate_fund(ladder, build_yield_curve(c['bills'], c['bonds']), max_maturity)
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

def simulate(max_maturity, min_maturity, country):
    """ This is just something to save on typing...and make clearer what the bounds on the fund are """
    # find the first non-NaN number in rates
    bonds = BOND_RATES[country].dropna()
    bills = BILL_RATES[country].dropna()
    
    start_year = 1970 #max(bills.head(1).index[0], bonds.head(1).index[0])
    if country == 'ESP': start_year = 1979

    end_year = 2017 #min(bills.tail(1).index[0], bonds.tail(1).index[0])
    
    rates = pandas.DataFrame.from_dict({'bills' : bills, 'bonds' : bonds})
    
    starting_rates = rates.loc[start_year]
    
    ladder = bootstrap(build_yield_curve(starting_rates['bills'], starting_rates['bonds']), max_maturity, min_maturity)
    return loop(ladder, rates, max_maturity, start_year, end_year)

simulate(10, 4, 'DNK').head()

countries = [
    'AUS',
    'AUT',
    'BEL',
    'CAN',
    'DNK',
    'FRA',
    'DEU',
    'ITA',
    'JPN',
    'NLD',
    'NOR',
#    'SGD', # 1999 onward
    'ESP', # 1979 onward
    'SWE',
    'CHE',
    'GBR',
    'USA',
    'ALL AVERAGE',
    '16 COUNTRIES AVERAGE',
    'NO JPN CHE AVERAGE',
]

pd = pandas.DataFrame(columns=countries)

for c in countries:
    print('Simulating ...', c)
    returns = simulate(10, 4, c)
    pd[c] = returns['Change']

pd.head()
pd.to_csv('oecd_10-4_returns.csv')



