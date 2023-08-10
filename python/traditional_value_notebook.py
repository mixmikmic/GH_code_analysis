import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.research import run_pipeline
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.factors import CustomFactor

# Custom Factor 1 : Dividend Yield
class Div_Yield(CustomFactor):

    inputs = [morningstar.valuation_ratios.dividend_yield]
    window_length = 1

    def compute(self, today, assets, out, d_y):
        out[:] = d_y[-1]

# create the pipeline
temp_pipe_1 = Pipeline()

# add the factor to the pipeline
temp_pipe_1.add(Div_Yield(), 'Dividend Yield')

# run the pipeline and get data for first 5 equities
run_pipeline(temp_pipe_1, start_date='2015-11-11', end_date='2015-11-11').head()

# Custom Factor 2 : P/B Ratio
class Price_to_Book(CustomFactor):

    inputs = [morningstar.valuation_ratios.pb_ratio]
    window_length = 1

    def compute(self, today, assets, out, pbr):
        out[:] = pbr[-1]
        
# create the Pipeline
temp_pipe_2 = Pipeline()

# add the factor to the Pipeline
temp_pipe_2.add(Price_to_Book(), 'P/B Ratio')

# run the Pipeline and get data for first 5 equities
run_pipeline(temp_pipe_2, start_date='2015-11-11', end_date='2015-11-11').head()

# Custom Factor 3 : Price to Trailing 12 Month Sales       
class Price_to_TTM_Sales(CustomFactor):
    inputs = [morningstar.valuation_ratios.ps_ratio]
    window_length = 1
    
    def compute(self, today, assets, out, ps):
        out[:] = -ps[-1]
        
# create the pipeline
temp_pipe_3 = Pipeline()

# add the factor to the pipeline
temp_pipe_3.add(Price_to_TTM_Sales(), 'Price / TTM Sales')

# run the pipeline and get data for first 5 equities
run_pipeline(temp_pipe_3, start_date='2015-11-11', end_date='2015-11-11').head()

# Custom Factor 4 : Price to Trailing 12 Month Cashflow
class Price_to_TTM_Cashflows(CustomFactor):
    inputs = [morningstar.valuation_ratios.pcf_ratio]
    window_length = 1
    
    def compute(self, today, assets, out, pcf):
        out[:] = -pcf[-1] 
        
# create the pipeline
temp_pipe_4 = Pipeline()

# add the factor to the pipeline
temp_pipe_4.add(Price_to_TTM_Cashflows(), 'Price / TTM Cashflows')

# run the pipeline and get data for first 5 equities
run_pipeline(temp_pipe_4, start_date='2015-11-11', end_date='2015-11-11').head()

# This factor creates the synthetic S&P500
class SPY_proxy(CustomFactor):
    inputs = [morningstar.valuation.market_cap]
    window_length = 1
    
    def compute(self, today, assets, out, mc):
        out[:] = mc[-1]
        
# Custom Factor 2 : P/B Ratio
class Price_to_Book(CustomFactor):

    inputs = [morningstar.valuation_ratios.pb_ratio]
    window_length = 1

    def compute(self, today, assets, out, pbr):
        out[:] = -pbr[-1]
        
def Data_Pull():
    
    # create the piepline for the data pull
    Data_Pipe = Pipeline()
    
    # create SPY proxy
    Data_Pipe.add(SPY_proxy(), 'SPY Proxy')

    # Div Yield
    Data_Pipe.add(Div_Yield(), 'Dividend Yield') 
    
    # Price to Book
    Data_Pipe.add(Price_to_Book(), 'Price to Book')
    
    # Price / TTM Sales
    Data_Pipe.add(Price_to_TTM_Sales(), 'Price / TTM Sales')
    
    # Price / TTM Cashflows
    Data_Pipe.add(Price_to_TTM_Cashflows(), 'Price / TTM Cashflow')
        
    return Data_Pipe

# NB: Data pull is a function that returns a Pipeline object, so need ()
results = run_pipeline(Data_Pull(), start_date='2015-11-11', end_date='2015-11-11')
results.head()

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
    
    # sort descending
    df_composite.sort(ascending=False)

    return df_composite

# compute the standardized values
results_standard, index = standard_frame_compute(results)

# aggregate the scores
ranked_scores = composite_score(results_standard, index)

# print the final rankings
ranked_scores

# create histogram of scores
ranked_scores.hist()

# make scores into list for ease of manipulation
ranked_scores_list = ranked_scores.tolist()

# add labels to axes
plt.xlabel('Standardized Scores')
plt.ylabel('Quantity in Basket')

# show long bucket
plt.axvline(x=ranked_scores_list[25], linewidth=1, color='r')

# show short bucket
plt.axvline(x=ranked_scores_list[-6], linewidth=1, color='r');



