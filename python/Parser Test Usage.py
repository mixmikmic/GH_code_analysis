from HASS_data_detective import DataParser 
from helpers import load_url
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

with open('C:/Users/Will Koehrsen/Desktop/db_url.txt', 'r') as f:
    db_url = f.read()

parser = DataParser(db_url)

parser.list_sensors

parser.all_corrs()

volume = pd.DataFrame(parser.get_sensors['sensor.volume_used_volume_1'])

plt.plot(volume)

phone = parser.get_sensors['sensor.robins_iphone_battery_level']

phone.head()

phone.tail()

model, future = parser.prophet_model(sensor='sensor.robins_iphone_battery_level', freq='M')

model.plot(future)

model.plot_components(future)



