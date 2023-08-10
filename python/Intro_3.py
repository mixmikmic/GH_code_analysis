import pandas_datareader.data as web

data = web.DataReader("IBM","google")

data.head()

data.head(10)

data.tail()

type(data)

data["Open"]

data["2015-05"]

data["2015-05":"2016-05"]

import matplotlib.pyplot as plt

# a sample plot of bisector line
plt.plot([1,2,3,4],[1,2,3,4])
plt.show()

plt.plot(data["High"])
plt.show()

data_apple = web.DataReader("AAPL",'google')

plt.plot(data["High"])
plt.plot(data_apple["High"])
plt.show()

