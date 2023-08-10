data.dtypes

data.set_index("Day Index").plot()

data.loc[(data['Sessions'] > 75000), 'Sessions'] = np.nan
data.loc[(data['Sessions'] < 8000), 'Sessions'] = np.nan
data.set_index('Day Index').plot();

data['Sessions'] = np.log(data['Sessions'])
data.set_index('Day Index').plot();

data.columns = ["ds", "y"]
data.head()

m1 = Prophet()
m1.fit(data)

future1 = m1.make_future_dataframe(periods=365)

forecast1 = m1.predict(future1)

forecast1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

forecast = np.exp(forecast1[['yhat', 'yhat_lower', 'yhat_upper']].tail())

m1.plot(forecast1);

m1.plot_components(forecast1);

m2 = Prophet()
m2.fit(data)
future2 = m2.make_future_dataframe(periods=90)
forecast2 = m2.predict(future2)
m2.plot(forecast2);

forecast1["Sessions"] = np.exp(forecast1.yhat).round()
forecast1["Sessions_lower"] = np.exp(forecast1.yhat_lower).round()
forecast1["Sessions_upper"] = np.exp(forecast1.yhat_upper).round()
forecast1[(forecast1.ds > "10-01-2017") &
          (forecast1.ds < "11-12-2017")][["ds", "yhat", "Sessions_lower", 
                                        "Sessions", "Sessions_upper"]]
next_year = forecast1[(forecast1.ds > "7-27-2017") &
          (forecast1.ds < "7-25-2018")][["ds", "yhat", "Sessions_lower", 
                                        "Sessions", "Sessions_upper"]]
next_year.max()

forecast1[(forecast1.ds > "7-27-2017") &
          (forecast1.ds < "7-25-2018")].plot(x="ds", y="Sessions", kind= "line")

#future2 = m2.make_future_dataframe(periods=90)
#forecast2= m2.predict(future2)
forecast2["Sessions"] = np.exp(forecast2.yhat).round()
forecast2["Sessions_lower"] = np.exp(forecast2.yhat_lower).round()
forecast2["Sessions_upper"] = np.exp(forecast2.yhat_upper).round()
forecast2[(forecast2.ds > "7-27-2017") &
          (forecast2.ds < "8-25-2017")][["ds", "yhat", "Sessions_lower", 
                                        "Sessions", "Sessions_upper"]]



