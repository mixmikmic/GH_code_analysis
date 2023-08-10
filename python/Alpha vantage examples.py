from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.sectorperformance import SectorPerformances
from alpha_vantage.cryptocurrencies import CryptoCurrencies
import matplotlib.pyplot as plt
import os

ts = TimeSeries(key=os.environ['API_KEY'], output_format='pandas')
data, meta_data = ts.get_intraday(symbol='MSFT',interval='1min', outputsize='compact')
# We can describe it
data.describe()

meta_data

data.plot(y=['open','close', 'high', 'low'])
plt.title('Intraday Times Series for the MSFT stock (1 min)')
plt.grid()
plt.show()

ti = TechIndicators(key=os.environ['API_KEY'], output_format='pandas')
data, meta_data = ti.get_bbands(symbol='MSFT', interval='60min', time_period=60)
data.describe()

meta_data

data.plot()
plt.title('BBbands indicator for  MSFT stock (60 min)')
plt.grid()
plt.show()

sp = SectorPerformances(key=os.environ['API_KEY'], output_format='pandas')
data, meta_data = sp.get_sector()
data.describe()

meta_data

data['Rank A: Real-Time Performance'].plot(kind='bar')
plt.title('Real Time Performance (%) per Sector')
plt.tight_layout()
plt.grid()
plt.show()

cc = CryptoCurrencies(key=os.environ['API_KEY'])
# There is no metadata in this call
data, _ = cc.get_currency_exchange_rate(from_currency='BTC',to_currency='USD')
data 

# I changed the internal format of the the class to be our friendly data frame.
cc.output_format='pandas'
data, meta_data = cc.get_digital_currency_intraday(symbol='BTC', market='CNY')
data.describe()

data.head(5)

data['. price (USD)'].plot()
plt.tight_layout()
plt.title('Intraday value for bitcoin (BTC)')
plt.grid()
plt.show()



