import pandas as pd

pd.__version__

my_series = pd.Series([4.6, 2.1, -4.0, 3.0])
my_series

my_series.values

hungarian_dictionary = {'spaceship': 'űrhajó'}

hungarian_dictionary['spaceship']

hungarian_dictionary = {'spaceship': 'űrhajó',
                        'watermelon': 'görögdinnye',
                        'bicycle': 'kerékpár'}

# Names (keys) mapped to a tuple (the value) containing the height, lat and longitude.
scottish_hills = {'Ben Nevis': (1345, 56.79685, -5.003508),
                  'Ben Macdui': (1309, 57.070453, -3.668262),
                  'Braeriach': (1296, 57.078628, -3.728024),
                  'Cairn Toul': (1291, 57.054611, -3.71042),
                  'Sgòr an Lochain Uaine': (1258, 57.057999, -3.725416)}

scottish_hills['Braeriach']

hills = pd.DataFrame(scottish_hills)
print(hills)

scottish_peaks = {'Hill Name': ['Ben Nevis', 'Ben Macdui', 'Braeriach', 'Cairn Toul', 'Sgòr an Lochain Uaine'],
                  'Height': [1345, 1309, 1296, 1291, 1258],
                  'Latitude': [56.79685, 57.070453, 57.078628, 57.054611, 57.057999],
                  'Longitude': [-5.003508, -3.668262, -3.728024, -3.71042, -3.725416]}

dataframe = pd.DataFrame(scottish_peaks)
print(dataframe)

dataframe = pd.DataFrame(scottish_peaks, columns=['Hill Name', 'Height', 'Latitude', 'Longitude'])
print(dataframe)

print(dataframe.head(3))

print(dataframe.tail(2))

print(dataframe['Hill Name'])

dataframe['Height']

#dataframe[0]

dataframe.iloc[0]

dataframe.iloc[0,0]

dataframe['Hill Name'][0]

dataframe.Height

dataframe.Height > 1300

dataframe[dataframe.Height > 1300]

dataframe['Region'] = ['Grampian', 'Cairngorm', 'Cairngorm', 'Cairngorm', 'Cairngorm']
print(dataframe)

dataframe = pd.read_csv("data/scottish_peaks.csv")

print(dataframe.head(10))

sorted_hills = dataframe.sort_values(by=['Height'], ascending=False)

# Let's have a look at the top 5 to check
print(sorted_hills.head(5))

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))

x = dataframe.Height
y = dataframe.Latitude

plt.scatter(x, y)
plt.savefig("python-scatter.png")

from scipy.stats import linregress

stats = linregress(dataframe['Height'], dataframe['Latitude'])

m = stats.slope
b = stats.intercept

#m, b = pd.np.polyfit(dataframe['Height'], dataframe['Latitude'], 1)
#plt.plot(dataframe['Height'], m * dataframe['Height'] + b)
plt.figure(figsize=(10,10))

# Change the default marker from circles to x's
plt.scatter(x, y, marker='x')

# Set the linewidth to 3px
plt.plot(x, m * x + b, color="red", linewidth=3)

# Add x and y lables, and set their font size
plt.xlabel("Height (m)", fontsize=20)
plt.ylabel("Latitude", fontsize=20)

# Set the font size of the number lables on the axes
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig("python-linear-reg-custom.png")

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature

plt.figure(figsize=(10,10))
ax = plt.axes(projection=ccrs.Mercator())
ax.coastlines('10m')

ax.xaxis.set_visible(True)
ax.yaxis.set_visible(True)

ax.set_yticks([56,57,58,59], crs=ccrs.PlateCarree())
ax.set_xticks([-8, -6, -4, -2], crs=ccrs.PlateCarree())

lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()

ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

ax.set_extent([-8, -1.5, 55.3, 59])

plt.scatter(dataframe['Longitude'],dataframe['Latitude'],
                    color='red', marker='^', transform=ccrs.PlateCarree())
plt.savefig("munros.png")

