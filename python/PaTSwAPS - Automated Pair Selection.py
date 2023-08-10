import math
import numpy as np
from datetime import date, timedelta
from statsmodels.tsa.stattools import coint
from quantopian.pipeline import CustomFactor, Pipeline
from quantopian.pipeline.factors import SimpleMovingAverage, AverageDollarVolume
from quantopian.pipeline.classifiers.morningstar import Sector
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.research import run_pipeline

# This function takes a dataframe of prices (price vs. time) and returns a list of any pairs that are cointegrated.

def find_cointegrated_pairs(data):    
    # Drop duplicated rows and set up necessary variables
    data = data.T.drop_duplicates().T
    m, n = data.shape[0], data.shape[1]
    pvalue_matrix = np.zeros((n, n))
    keys = data.keys()
    pairs = []
    
    # Make a matrix of p-values
    for i in range(n):
        for j in range(i+1, n):
            result = coint(data[keys[i]], data[keys[j]])
            pvalue_matrix[i, j] = result[1]
    
    # Find uniquely cointegrated pairs of securities
    alpha = sidak(0.05, n*(n-1)/2)
    for i in range(n):
        for j in range(i+1, n):
            check1 = (pvalue_matrix[k, j] >= 0.5 for k in range(i))
            check2 = (pvalue_matrix[i, k] >= 0.5 for k in range(j, n))
            #check3 = (not math.isnan(float(pvalue_matrix[k, j])) for k in range(i))
            #check4 = (not math.isnan(float(pvalue_matrix[i, k])) for k in range(j, n))
            if (pvalue_matrix[i, j] <= alpha) and check1 and check2: #and check3 and check4:
                pairs.append((keys[i].symbol, keys[j].symbol))
    
    return pairs

def sidak(fwer, num_comps):
    return np.float128(1-(1-fwer)**(1.0/num_comps))

# Interesting cointegrated pairs to keep track of!
foo = get_pricing(['CSUN', 'ASTI', 'ABGB', 'FSLR'], '01-01-2014', '01-01-2015', fields='price')
find_cointegrated_pairs(foo)

# Define and instantiate all necessary factors

class Market_Cap(CustomFactor):
    inputs = [morningstar.valuation.market_cap]
    window_length = 1
    def compute(self, today, assets, out, inputs):
        out[:] = inputs

class Industry_Group(CustomFactor):
    inputs = [morningstar.asset_classification.morningstar_industry_group_code]
    window_length = 1
    def compute(self, today, assets, out, inputs):
        out[:] = inputs

avg_close = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=20)
avg_vol = AverageDollarVolume(window_length=20)
sector = Sector()
group = Industry_Group()
market_cap = Market_Cap()

# The Pipeline filters the universe by group code, and applies minimum acceptance requirements (enumerated below)

def make_pipeline(group_code):
    sector_filter = sector.notnull() # No stocks in misc. sector
    penny_stock_filter = (avg_close > 5.0) # No stocks that are less that $5
    volume_filter = (avg_vol > 750000) # No companies who have a dollar volume of less than $0.75m
    small_cap_filter = (market_cap >= 300000000) # No companies who are valued at less than $300m
    group_filter = group.eq(group_code) # No companies that are not in the industry group under consideration
    
    return Pipeline(
        columns = {'industry_group':group},
        screen = (sector_filter & penny_stock_filter & volume_filter & small_cap_filter & group_filter)
    )

# Morningstar industry group codes: for mappings, see https://www.quantopian.com/help/fundamentals#industry-sector

group_codes = [10101, 10102, 10103, 10104, 10105, 10106, 10107,
               10208, 10209, 10210, 10211, 10212, 10213, 10214, 10215, 10216, 10217, 10218,
               10319, 10320, 10321, 10322, 10323, 10324, 10325, 10326,
               10427, 10428,
               20529, 20530, 20531, 20532, 20533, 20534,
               20635, 20636, 20637, 20638, 20639, 20640, 20641, 20642,
               20743, 20744,
               30845,
               30946, 30947, 30948, 30949, 30950, 30951,
               31052, 31053, 31054, 31055, 31056, 31057, 31058, 31059, 31060, 31061, 31062, 31063, 31064,
               31165, 31166, 31167, 31168, 31169]

# This code goes through accepted stocks in each industry group and find if any stocks are cointegrated over the
# past 365 days

pairs = []
start = '01-01-2014'
end = '01-01-2015'

for i in range(len(group_codes)):
    symbols = []
    pipe_output = run_pipeline(make_pipeline(group_codes[i]), end, end)
    for j in range(len(pipe_output.index)):
        symbols.append(pipe_output.index.values[j][1].symbol)
    if symbols != []:
        prices = get_pricing(symbols, start, end, fields='price')
        prices.dropna(axis=1)
        pairs = pairs + find_cointegrated_pairs(prices)

pairs

pairs = []
start = '01-01-2014'
end = '01-01-2015'

i = 19
symbols = []
pipe_output = run_pipeline(make_pipeline(group_codes[i]), end, end)
for j in range(len(pipe_output.index)):
    symbols.append(pipe_output.index.values[j][1].symbol)
if symbols != []:
    prices = get_pricing(symbols, start, end, fields='price')
    prices.dropna(axis=1)
    pairs = pairs + find_cointegrated_pairs(prices)

pairs



