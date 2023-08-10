get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
from fbprophet import Prophet
from matplotlib import pyplot as plt
import time
from datetime import datetime as dt
from datetime import timedelta as td

class Predictor():
    def __init__(self, currency, training_days, test_days, changepoint_prior_scale=1):
        self.currency = currency
        self.training_days = training_days
        self.test_days = test_days
        self.changepoint_prior_scale = changepoint_prior_scale
        self.model = Prophet(changepoint_prior_scale=changepoint_prior_scale, yearly_seasonality=False, daily_seasonality=False)
        assert type(self.currency) == str
    
    def __len__(self):
        return self.training_days + self.test_days

    def make_model(self):
        
        # Set date ranges to collect training and test data from CoinMarketCap
        test_end = dt.today()
        test_start = test_end - td(days=self.test_days-1)
        training_end = test_start - td(days=1)
        training_start = test_start - td(days=self.training_days)

        test_end = test_end.strftime("%Y%m%d")
        test_start = test_start.strftime("%Y%m%d")
        training_end = training_end.strftime("%Y%m%d")
        training_start = training_start.strftime("%Y%m%d") 

        # Read current price from CMC api
        market_info_for_today = pd.read_json("https://api.coinmarketcap.com/v1/ticker/" + self.currency)
        market_info_for_today.drop(market_info_for_today.columns[[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]], axis=1, inplace=True)
        market_info_for_today['last_updated'] = market_info_for_today['last_updated'].apply(lambda x: dt.fromtimestamp(int(x)))
        market_info_for_today.columns = ['ds', 'y']
        
        # Read table from historical data page of cryptocurrency
        if self.test_days > 1:
            market_info_test = pd.read_html("https://coinmarketcap.com/currencies/" + self.currency + "/historical-data/?start=" + test_start + "&end=" + test_end)[0]
            market_info_test.drop(market_info_test.columns[[1, 2, 3, 5, 6]], axis=1, inplace=True)
            market_info_test = market_info_test.assign(Date=pd.to_datetime(market_info_test['Date']))
            market_info_test.iloc[::-1].sort_index(axis=0 ,ascending=False)
            market_info_test.columns = ['ds', 'y']
            market_info_test = pd.concat([market_info_for_today, market_info_test], axis=0, ignore_index=True)
        else:
            market_info_test = market_info_for_today
        
        market_info_test['y'] = np.log(market_info_test['y'])
        self.test = market_info_test.iloc[::-1].sort_index(axis=0 ,ascending=False)

        market_info_training = pd.read_html("https://coinmarketcap.com/currencies/" + self.currency + "/historical-data/?start=" + training_start + "&end=" + training_end)[0]
        market_info_training.drop(market_info_training.columns[[1, 2, 3, 5, 6]], axis=1, inplace=True)
        market_info_training = market_info_training.assign(Date=pd.to_datetime(market_info_training['Date']))
        market_info_training.columns = ['ds', 'y']
        market_info_training['y'] = np.log(market_info_training['y'])
        self.training = market_info_training.iloc[::-1].sort_index(axis=0 ,ascending=False)
        
        
        # Fit training data to model and eval speed
        print("Fitting training data...")
        start = time.time()
        self.model.fit(self.training)
        print("Took", round(time.time()-start, 4), "seconds.", "\n")
        
        
        # Evaluate model by calculating Mean Absolute Error of predictions 
        # for test days with actual prices on those days
        print("Evaluating model on testing data...")
        future = self.model.make_future_dataframe(periods=self.test_days)
        forecast = self.model.predict(future)
        forecast = forecast[-self.test_days:]
        
        revert = lambda x: np.e**x
        forecast['yhat'] = forecast['yhat'].apply(revert)
        forecast['yhat_upper'] = forecast['yhat_upper'].apply(revert)
        forecast['yhat_lower'] = forecast['yhat_lower'].apply(revert)
        self.error_forecast = forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']]
        test = self.test.copy(deep=True)['y'].apply(revert)
        
        if not all([x.date()==y.date() for x, y in zip(forecast['ds'], self.test['ds'])]):
            raise ValueError("Test or Forecast Dataframe isn't ordered by date or order is mismatched")
        self.mae_upper = sum([abs(yhat - y) for yhat, y in zip(forecast['yhat_upper'], test)]) / len(forecast)
        self.mae = sum([abs(yhat - y) for yhat, y in zip(forecast['yhat'], test)]) / len(forecast)
        self.mae_lower = sum([abs(yhat - y) for yhat, y in zip(forecast['yhat_lower'], test)]) / len(forecast)
        print("Upper MAE: $" + str(self.mae_upper))
        print("MAE: $" + str(self.mae))
        print("Lower MAE: $" + str(self.mae_lower) + "\n")
    
    def predict(self, days=30, make_plot=True):
        # Reinitialize model and fit both training and test 
        # data to create most up-to-date model
        self.model = Prophet(changepoint_prior_scale=self.changepoint_prior_scale, yearly_seasonality=False, daily_seasonality=False)
        print("Fitting all data...")
        start = time.time()
        self.model.fit(pd.concat([self.test, self.training]))
        print("Took", round(time.time()-start, 3), "seconds.")
        
        future = self.model.make_future_dataframe(periods=days)
        self.forecast = self.model.predict(future)
        
        revert = lambda x: np.e**x
        self.model.history['y'] = self.model.history['y'].apply(revert)
        self.forecast['yhat'] = self.forecast['yhat'].apply(revert)
        self.forecast['yhat_upper'] = self.forecast['yhat_upper'].apply(revert)
        self.forecast['yhat_lower'] = self.forecast['yhat_lower'].apply(revert)
        
        # Plot forecast and actual data if plotting desired
        if make_plot: return self.model.plot(self.forecast), self.model.plot_components(self.forecast)
        

# Init predictor object for ethereum with 150 training days and 3 test days. 
# The testing period is short because thats the max range of days that the model
# is accurate enough for. As such, this model should not be used for long forecasting 
# of prices and should only be used for finding short-term prices or the long-term 
# trend (up or down).
ether_predictor = Predictor("ethereum", 150, 3)

# Assemble model
ether_predictor.make_model()

# Predict the next 15 days worth of prices
ether_predictor.predict(15)

# The forecast for testing period
ether_predictor.error_forecast.head()

# Actual value(s) for testing period
ether_predictor.test['y'] = ether_predictor.test['y'].apply(lambda x: np.e**x)
ether_predictor.test.head()

# Forecast for the next 15 days
ether_predictor.forecast[['ds', 'yhat']].tail(15)

