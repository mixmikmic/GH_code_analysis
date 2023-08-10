import pandas as pd
import numpy as np
import warnings
# Dickey-Fuller Test
from statsmodels.tsa.stattools import adfuller
# auto corelation
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
# seasonal decompose
from statsmodels.tsa.seasonal import seasonal_decompose
# ARIMA
import statsmodels.tsa as tsa
# display
from IPython.display import display

warnings.filterwarnings('ignore')

# plot utilities
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import mpld3
mpld3.enable_notebook()

def testStationarity(ts=None, displaySummary=True, displayPlot=True):
    ''' Test whether the input series is stationary or not
    '''
    # remove NAN's
    ts = ts.dropna()
    
    # create a data frame for X1
    dframe = pd.DataFrame(ts.index)
    dframe['P(t)'] = ts.values
    d = dframe['P(t)'][~dframe['P(t)'].isnull()]

    # dickyey-fuller test
    df = adfuller(ts)
    if df[0] < df[4]['1%']:
        confi = 0.99
        isStationary = True;
        strStationary = ' DFTest: Stationary'+ ' (confidence= %.2f)' % confi
        
    elif df[0] < df[4]['5%']:
        confi = 0.95
        isStationary = True;
        strStationary = ' DFTest: Stationary'+ ' (confidence= %.2f)' % confi
        
    elif df[0] < df[4]['10%']:
        confi = 0.90
        isStationary = True;
        strStationary = ' DFTest: Stationary' + ' (confidence= %.2f)' % confi
        
    else:
        confi = 0
        isStationary = False;
        strStationary = ' DFTest: Non Stationary'
        
    # moving averages
    dframe['MA_30'] = pd.Series.rolling(d, center=False, window=30).mean() # 30 -day MA
    dframe['MA_60'] = pd.Series.rolling(d, center=False, window=60).mean() # 60 -day MA
    dframe['MA_120'] = pd.Series.rolling(d, center=False, window=120).mean() # 120 -day MA

    # standard devs
    dframe['SD_30'] = pd.Series.rolling(d, center=False, window=30).std() # 30 -day SD
    dframe['SD_60'] = pd.Series.rolling(d, center=False, window=60).std() # 60 -day SD
    dframe['SD_120'] = pd.Series.rolling(d, center=False, window=120).std() # 120 -day SD
    
    if displayPlot:
        # plot
        dframe.loc[:, ['MA_30', 'MA_60', 'MA_120']].plot( title='Moving Averages - ' + ts.name + strStationary)
        dframe.loc[:, ['SD_30', 'SD_60', 'SD_120']].plot( title='Variance - ' + ts.name + strStationary)
        
    if displaySummary:
        print(df)
        print('Time series - '+ts.name + ',' + strStationary)
        
    return isStationary


def convert2Stationary(ts=None, disp=True):
    ''' Convert a non-stationary time-series to stationary using simple differencing of
        log-transformed timeseries
        The input time-series is first checked for stationarity then the conversion is done
        
        input: ts -> time-series in normal price
        output: stationary timeseries
    '''
    # log
    ts_log = np.log(ts.dropna())
    out = ts_log
    if not testStationarity(ts, displayPlot=False, displaySummary=False):
        # differencing
        out = ts_log - ts_log.shift(1)
    
    # perform test
    if disp:
        print("=== after conversion test for Stationarity===")
        testStationarity(out, displayPlot=disp, displaySummary=disp)
        
    return out



def forecast_3day_returns(ts=None, ARIMAorder=None):
    ''' Forecast 3-day returns using ARIMA modelling
    '''
    # original timeseries
    orig_tseries = ts
    # create a new time series by pushing it 3-days ahead and fill it with 0
    pushed_tseries = pd.Series(0, orig_tseries.index + pd.Timedelta('3 day'))
    # extract the last 3-days from pushed_tseries and append it to original timeseries
    tseries = orig_tseries.append(pushed_tseries[-3 : pushed_tseries.shape[0]], verify_integrity=True)

    # convert original timeseries X3 to stationary
    day_returns = convert2Stationary(ts, False)
    # index-0 of day_returns will be NAN due to differencing.
    # usually day_returns[original_length+1] will have -INF due to 0 insertion.

    if ARIMAorder is None:
        ARIMAorder = (1,0,1)
        
    # modelling (use all data starting from index-1 day of day_returns to the lastday of the orignal timeseries)
    model = tsa.arima_model.ARIMA(day_returns.ix[1:], ARIMAorder).fit(disp=-1)
    print(model.summary())

    # index of day_returns runs from 0:day_returns.shape[0]-1, there are total of day_returns.shape[0] elements
    # we are ignoring inedx-0 while training (it might contain NAN due to differencing), hence we are only training
    # with day_returns.shape[0]-1 elements (with indexing running from 1:day_returns.shape[0]-1). 
    # Therefore the the trained-model will have a length of ((day_returns.shape[0]-1)-1)
    # For prediction, the starting index should atleast be the last fittedvalue.

    # predict
    preds = model.predict(model.fittedvalues.shape[0]-1, model.fittedvalues.shape[0]+2 )

    ### calculate error metrics and accuracy of fittedvalues
    # Original-returns: calculate rolling_sum of log-returns over a window of 4 days, including current day
    orig_returns = day_returns[1:]
    rollsum = pd.Series.rolling(orig_returns, center=False, window=4).sum()
    orig_3day_returns =  orig_returns - rollsum.shift(-3)
    # fitted-returns: 
    rollsum = pd.Series.rolling(model.fittedvalues, center=False, window=4).sum()
    fitted_3day_returns =  model.fittedvalues - rollsum.shift(-3)
    # display dataframe

    statsFrame = pd.DataFrame(orig_returns.index)
    statsFrame['Orig Log Returns'] = orig_returns.values
    statsFrame['Pred Log Returns'] = model.fittedvalues.values
    statsFrame['AbsError'] = np.abs(orig_returns.values - model.fittedvalues.values)
    statsFrame['AbsPercentError(%)'] = np.abs((orig_returns.values - model.fittedvalues.values)/(orig_returns.values+0.00000001)) * 100 
    statsFrame['DirectionAccuracy'] = ~((orig_returns.values > 0) != (model.fittedvalues.values > 0))
    statsFrame['Orig 3-day Log returns'] = orig_3day_returns.values
    statsFrame['Pred 3-day Log returns'] = fitted_3day_returns.values
    statsFrame['3-day AbsError'] = np.abs(orig_3day_returns.values - fitted_3day_returns.values)
    statsFrame['3-day AbsPercentError(%)'] = np.abs((orig_3day_returns.values - fitted_3day_returns.values)/(orig_3day_returns.values+0.00000001)) * 100 
    statsFrame['3-day DirectionAccuracy'] = ~((orig_3day_returns.values > 0) != (fitted_3day_returns.values > 0))
    # calculate


    # create a fitted_day_returns series using index from new timeseries (created used pushed_timeseries) and fittedvalues
    fitted_day_returns = pd.Series(model.fittedvalues, index=tseries.index)
    # copy the forecast values to correct dates
    fitted_day_returns[-4 : fitted_day_returns.shape[0]] = preds
    forecast_day_returns = fitted_day_returns[-4 : fitted_day_returns.shape[0]]

    # Predicted-returns: 
    rollsum = pd.Series.rolling(forecast_day_returns, center=False, window=4).sum()
    forecast_3day_returns =  forecast_day_returns - rollsum.shift(-3)

    # Display
    print("=====  3-day Returns forecast for closing price of last-day in timeseries ======")
    print("           Note: The first row in the table corresponds to the last day \n           closing price of the stock")
    print("   Stock: "+ ts.name)
    futureFrame = pd.DataFrame(forecast_day_returns.index)
    futureFrame['Forecast Log Returns'] = forecast_day_returns.values
    futureFrame['Forecast 3-day Log Returns'] = forecast_3day_returns.values
    futureFrame['Long term MAE (log returns)'] = np.mean(statsFrame['AbsError'])
    futureFrame['Long term MAPE (log returns) %'] = np.mean(statsFrame['AbsPercentError(%)']/100)*100
    futureFrame['Long term DirectionAccuracy (log returns) %'] = (sum(statsFrame['DirectionAccuracy'])/statsFrame.shape[0]) * 100
    futureFrame['Long term MAE (3-day log returns)'] = np.mean(statsFrame['3-day AbsError'])
    futureFrame['Long term MAPE (3-day log returns) %'] = np.mean(statsFrame['3-day AbsPercentError(%)']/100) * 100 
    futureFrame['Long term DirectionAccuracy (3-day log returns) %'] = (sum(statsFrame['3-day DirectionAccuracy'])/statsFrame.shape[0]) * 100
    display(futureFrame)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    
    
    return futureFrame



# test- forecast
# convert data to time-series data while reading from csv itself
dailyPrices = pd.read_csv('dailyPrices_AtClose.csv', parse_dates='t()', index_col='t()')
#dailyPrices.index # check if the index is datetime
# re-index and set frequency='D'
dates = pd.date_range(dailyPrices.index[0], dailyPrices.index[-1], freq='D')
dailyPrices = dailyPrices.reindex(dates)
#dailyPrices.index

forecast_3day_returns(dailyPrices.X3)
    

# process
for stock in dailyPrices:
    print('*******************************    Stock :  ' + dailyPrices[stock].name + '   ****************************')
    forecast_3day_returns(dailyPrices[stock], ARIMAorder=(1,0,0))
    



