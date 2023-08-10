import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quantopian.pipeline.classifiers.morningstar import Sector
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.research import run_pipeline
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.factors import CustomFactor

# Custom Factor 1 : Price to Trailing 12 Month Sales       
class Price_to_TTM_Sales(CustomFactor):
    inputs = [morningstar.valuation_ratios.ps_ratio]
    window_length = 1
    
    def compute(self, today, assets, out, ps):
        out[:] = ps[-1]

# create pipeline
temp_pipe_1 = Pipeline()

# add our factor
temp_pipe_1.add(Price_to_TTM_Sales(), 'Price / TTM Sales')

# add sector classifier
temp_pipe_1.add(Sector(), 'Sector')

# get data
temp_res_1 = run_pipeline(temp_pipe_1, '2015-06-06', '2015-06-06')

# show first 15 rows
temp_res_1.head(15)

# Separate sectors into two data frames
retail_df = temp_res_1[temp_res_1['Sector'] == 102]['Price / TTM Sales']
tech_df = temp_res_1[temp_res_1['Sector'] == 311]['Price / TTM Sales']

# get quartiles and print results
print 'Retail Quartiles: \n' + str(retail_df.quantile([0.25, 0.5, 0.75])) + '\n'
print 'Tech Quartiles: \n' + str(tech_df.quantile([0.25, 0.5, 0.75]))

# Custom Factor 2 : Price to Trailing 12 Month Earnings
class Price_to_TTM_Earnings(CustomFactor):
    inputs = [morningstar.valuation_ratios.pe_ratio]
    window_length = 1
    
    def compute(self, today, assets, out, pe):
        out[:] = pe[-1]
        
temp_pipe_2 = Pipeline() 
temp_pipe_2.add(Price_to_TTM_Earnings(), 'Price / TTM Earnings')
temp_pipe_2.add(Sector(), 'Sector')
temp_res_2 = run_pipeline(temp_pipe_2, '2015-06-06', '2015-06-06')

# clean extreme data points
earnings_frame = temp_res_2[temp_res_2['Price / TTM Earnings'] < 100]

# create boxplot by sector
earnings_frame.boxplot(column='Price / TTM Earnings', by='Sector');

# Custom Factor 3 : Price to Trailing 12 Month Cash Flows
class Price_to_TTM_Cashflows(CustomFactor):
    inputs = [morningstar.valuation_ratios.pcf_ratio]
    window_length = 1
    
    def compute(self, today, assets, out, pcf):
        out[:] = pcf[-1] 
        
temp_pipe_3 = Pipeline() 
temp_pipe_3.add(Price_to_TTM_Cashflows(), 'Price / TTM Cashflows')
temp_pipe_3.add(Sector(), 'Sector')
temp_res_3 = run_pipeline(temp_pipe_3, '2015-06-06', '2015-06-06')

# clean extreme data points
cashflows_frame = temp_res_3[temp_res_3['Price / TTM Cashflows'] < 100]

# create boxplot by sector
cashflows_frame.boxplot(column='Price / TTM Cashflows', by='Sector');

# clean data, necessary as using mean and standard deviation
retail_df = retail_df[retail_df < 10]
tech_df = tech_df[tech_df < 10]

# summary stats necessary for calculation
retail_mean = retail_df.mean()
retail_std = retail_df.std()
tech_mean = tech_df.mean()
tech_std = tech_df.std()

# standardize the data
retail_standard = (retail_df - retail_mean) / retail_std
tech_standard = (tech_df - tech_mean) / tech_std

# create a grid of plots
fig, axes = plt.subplots(nrows=2, ncols=2)
# name each set of axes
ax_retail, ax_tech, ax_retail_st, ax_tech_st = axes.flat
# number of bins for histograms
bins = 50

# retail
ax_retail.hist(retail_df, bins=bins)
ax_retail.axvline(retail_mean, color='blue')
ax_retail.axvline(retail_mean - retail_std, color='blue')
ax_retail.axvline(retail_mean + retail_std, color='blue')
ax_retail.set_xlabel('Price / Sales')
ax_retail.set_ylabel('Frequency')
ax_retail.set_title('Retail')

# tech
ax_tech.hist(tech_df, bins=bins, stacked=True)
ax_tech.axvline(tech_mean, color='green')
ax_tech.axvline(tech_mean - tech_std, color='green')
ax_tech.axvline(tech_mean + tech_std, color='green')
ax_tech.set_xlabel('Price / Sales')
ax_tech.set_ylabel('Frequency')
ax_tech.set_title('Technology')

# retail standardized
ax_retail_st.hist(retail_standard, bins=bins)
ax_retail_st.axvline(0, color='blue')
ax_retail_st.axvline(-1, color='blue')
ax_retail_st.axvline(1, color='blue')
ax_retail_st.set_xlabel('Standard Deviations')
ax_retail_st.set_ylabel('Frequency')
ax_retail_st.set_title('Retail Standard')

# tech standardized
ax_tech_st.hist(tech_standard, bins=bins, stacked=True)
ax_tech_st.axvline(0, color='green')
ax_tech_st.axvline(-1, color='green')
ax_tech_st.axvline(1, color='green')
ax_tech_st.set_xlabel('Standard Deviations')
ax_tech_st.set_ylabel('Frequency')
ax_tech_st.set_title('Technology Standard')

# prevent text overlap
fig.tight_layout()

# This factor creates the synthetic S&P500
class SPY_proxy(CustomFactor):
    inputs = [morningstar.valuation.market_cap]
    window_length = 1
    
    def compute(self, today, assets, out, mc):
        out[:] = mc[-1]
        
        
# this function returns a pipeline that downloads all data necessary for the algo
def Data_Pull():

    # create the piepline for the data pull
    Data_Pipe = Pipeline()
    
    # create sector partitions
    sector = Sector()
    
    # create SPY proxy
    Data_Pipe.add(SPY_proxy(), 'SPY Proxy')
    
    # Price / TTM Sales grouped by Industry
    sales_grouped = Price_to_TTM_Sales().zscore(groupby=sector)
    
    # Price / TTM Earnings grouped by Industry
    earnings_grouped = Price_to_TTM_Earnings().zscore(groupby=sector)
    
    # Price / TTM Cashflows grouped by Industry
    cashflows_grouped = Price_to_TTM_Cashflows().zscore(groupby=sector)
    
    # add Price / TTM Sales to Pipeline
    Data_Pipe.add(sales_grouped, 'Price / TTM Sales')
    
    # add Price / TTM Earnings to Pipeline
    Data_Pipe.add(earnings_grouped, 'Price / TTM Earnings')
    
    # add Price / TTM Cashflows to Pipeline
    Data_Pipe.add(cashflows_grouped, 'Price / TTM Cashflows')
    
    return Data_Pipe

results = run_pipeline(Data_Pull(), '2015-06-06', '2015-06-06')
results.head(20)

# limit effect of outliers
def filter_fn(x):
    if x <= -10:
        x = -10.0
    elif x >= 10:
        x = 10.0
    return x   

# standardize using mean and sd of S&P500
def standard_frame_compute(df):
    
    # basic clean of dataset to remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # need standardization params from synthetic S&P500
    df_SPY = df.sort(columns='SPY Proxy', ascending=False)

    # create separate dataframe for SPY
    # to store standardization values
    df_SPY = df_SPY.head(500)
    
    # get dataframes into numpy array
    df_SPY = df_SPY.as_matrix()
    
    # store index values
    index = df.index.values
    df = df.as_matrix()
    
    df_standard = np.empty(df.shape[0])
    
    
    for col_SPY, col_full in zip(df_SPY.T, df.T):
        
        # summary stats for S&P500
        mu = np.mean(col_SPY)
        sigma = np.std(col_SPY)
        col_standard = np.array(((col_full - mu) / sigma)) 

        # create vectorized function (lambda equivalent)
        fltr = np.vectorize(filter_fn)
        col_standard = (fltr(col_standard))
        
        # make range between -10 and 10
        col_standard = (col_standard / df.shape[1])
        
        # attach calculated values as new row in df_standard
        df_standard = np.vstack((df_standard, col_standard))
     
    # get rid of first entry (empty scores)
    df_standard = np.delete(df_standard,0,0)
    
    return (df_standard, index)

# Sum up and sort data
def composite_score(df, index):

    # sum up transformed data
    df_composite = df.sum(axis=0)
    
    # put into a pandas dataframe and connect numbers
    # to equities via reindexing
    df_composite = pd.Series(data=df_composite,index=index)
    
    # sort ascending - change from previous notebook
    df_composite.sort(ascending=True)

    return df_composite

# compute the standardized values
results_standard, index = standard_frame_compute(results)

# aggregate the scores
ranked_scores = composite_score(results_standard, index)
ranked_scores

# create the histogram
ranked_scores.hist(bins=50)

# make scores into list for ease of manipulation
ranked_scores_list = ranked_scores.tolist()

# add labels to axes
plt.xlabel('Standardized Scores')
plt.ylabel('Quantity in Basket')

# show long bucket
plt.axvline(x=ranked_scores_list[25], linewidth=1, color='r')

# show short bucket
plt.axvline(x=ranked_scores_list[-6], linewidth=1, color='r');

# create Pipeline for sectors
sector_pipe = Pipeline()
sector_pipe.add(Sector(), 'Sector')
sectors = run_pipeline(sector_pipe, '2015-06-06', '2015-06-06')

# connect ranked scores with their sectors
scores_sectors = pd.concat([ranked_scores, sectors], axis=1, join='inner')

# name the columns
scores_sectors.columns=['Score', 'Sector']

# sort ranked scores ascending
scores_sectors.sort('Score', inplace=True)

# show long bucket
scores_sectors.head(26)

SECTOR_NAMES = {
 101: 'Basic Materials',
 102: 'Consumer Cyclical',
 103: 'Financial Services',
 104: 'Real Estate',
 205: 'Consumer Defensive',
 206: 'Healthcare',
 207: 'Utilities',
 308: 'Communication Services',
 309: 'Energy',
 310: 'Industrials',
 311: 'Technology' ,
}

# create and populate the buckets
long_bucket = pd.Series()
short_bucket = pd.Series()
for key in SECTOR_NAMES:
    long_bucket = long_bucket.append(scores_sectors[scores_sectors['Sector'] == key]['Score'].head(13))
    short_bucket = short_bucket.append(scores_sectors[scores_sectors['Sector'] == key]['Score'].tail(3))        

print 'LONG BUCKET\n' + str(long_bucket) + '\n'
print 'SHORT BUCKET\n' + str(short_bucket)

