# Our boilerplate imports
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import numpy as np
import pandas as pd

global_co2_emission = pd.read_csv('sample_datasets/global_co2_emission.csv')

fig = plt.figure()
ax = fig.add_subplot(111) 
theta = np.linspace(-np.pi, np.pi, 50)
plt.scatter(theta, np.sin(theta))

fig = plt.figure()
ax = fig.add_subplot(111) 
theta = np.linspace(-np.pi, np.pi, 50)
plt.scatter(theta, np.sin(theta), marker="*")

N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii
fig = plt.figure()
plt.scatter(x, y, s=area, c=colors, alpha=0.5, cmap='gist_earth')
plt.colorbar()

plt.bar(np.arange(10), np.random.randint(1, 100, 10), yerr=np.random.randint(2, 10, 10))

fig, ax = plt.subplots(figsize=(15,7))
ax.bar(global_co2_emission['Year'], global_co2_emission['Total'], 
        width=0.5, align='center', color='#34495e')
ax.set_xticks(global_co2_emission['Year'][::10]);
ax.set_xlabel('Year')
ax.set_ylabel('Million Tons of $CO_2$')
ax.set_title('Global $CO_2$ emissions')

bars = plt.bar([1, 2, 3, 4], [10, 12, 15, 17])
# add hatches to the first bar
plt.setp(bars[0], hatch='x-',)
plt.show()

labels       = ['Samsung', 'Apple', 'Huawei', 'Oppo', 'Vivo', 'Others']
market_share = [21, 12.5, 9.3, 7.1, 5.9, 44.2]
explode      = [0.2, 0, 0, 0, 0, 0]
fig, ax      = plt.subplots()
ax.pie(market_share, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio so that we get a circle
ax.axis('equal');

#Random numbers from normal distribution
data = np.random.randn(1000)
print("Max: ", data.max())
print("Min: ", data.min())
fig, ax = plt.subplots()
ax.hist(data)

fig, ax = plt.subplots()
ax.hist(data, bins=30, normed=True, alpha=0.8,
         histtype='barstacked', color='lightcoral',
         edgecolor='none');

import scipy.misc
racoon  = scipy.misc.face()
fig, ax = plt.subplots()
ax.imshow(racoon)
# disable the grid lines and axis
ax.grid(False)
ax.axis('off')

# The image is a 3-D numpy array.. 3D = Red, Green, Blue
print('Type: ', type(racoon))
# An RGB image should have a shape of rows x cols x 3
print("Shape: ", racoon.shape)
# Lets check the dtype
print("Dtype: ", racoon.dtype)
# dype of uint8 means a range of 0 - 255

# Great, now lets plot the red, green and blue color channels
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
colors = ['red', 'green', 'dodgerblue']
for i in range(3):
    axes[i][0].imshow(racoon[:,:,i])
    axes[i][1].hist(racoon[:,:,i].ravel(), bins=255, alpha=0.3, color=colors[i])

# For Red channel
counts, bins = np.histogram(racoon[:,:,0], bins=255)
print("Counts: ", counts)

