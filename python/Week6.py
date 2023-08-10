import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (15, 9)

data = pd.read_csv('^GSPC.csv')

data.head()

data['Date'] = [datetime.datetime.strptime(data['Date'][i], '%m/%d/%y').                strftime('%Y-%m-%d') for i in range(len(data['Date']))]

data.describe() #analogous to R's summary(df1) function

data['Adj Close'].plot(grid=True)
plt.xlabel("Observations")
plt.ylabel("Price")
plt.title("S&P 500 Adjusted Close")

returns = data['Adj Close'].pct_change(1)
returns.plot(grid=True)
plt.ylabel("Returns")
plt.title("S&P 500 Returns")

sentiment = pd.read_csv("sentiment_daily.csv")

sentiment.head()

sentiment = sentiment[sentiment['year'] >= 1990]
sentiment  = sentiment[['Date', 'score']]

sentiment.head()

sentiment = sentiment.interpolate()

plt.plot(sentiment['score'])

df = pd.merge(sentiment, data, on = 'Date')
df.head()

plt.scatter(df['score'], df['Adj Close'].pct_change(1))
plt.ylabel("Returns")
plt.xlabel("Sentiment Score")

volatility = df['Adj Close'].rolling(window = 30).std()
volatility.head()

plt.plot(volatility)

plt.scatter(df['score'], volatility)
plt.xlabel("Score")
plt.ylabel("Volatility")
plt.title("Sentiment versus Volatility (S&P 500) from 1990 - 2017")

df.to_csv("filename.csv")

