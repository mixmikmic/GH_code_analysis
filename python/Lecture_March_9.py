import bs4

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

driver = webdriver.Chrome()

driver.get('https://www.google.com/flights/explore/')

to_input = driver.find_element_by_xpath('//*[@id="root"]/div[3]/div[3]/div/div[4]/div/div')

to_input.click()

actions = ActionChains(driver)
actions.send_keys('South America')
actions.send_keys(Keys.ENTER)
actions.perform()

results = driver.find_elements_by_class_name('LJTSM3-v-d')
test = results[0]
bars = test.find_elements_by_class_name('LJTSM3-w-x')
print len(results)
results[:2]

test.text

import time
data = []

for bar in bars:
    ActionChains(driver).move_to_element(bar).perform()
    time.sleep(0.001)
    data.append((test.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[0].text,
           test.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[1].text))

print data[:5]
data[-5:]

import pandas as pd
import datetime
get_ipython().magic('matplotlib inline')
from dateutil.parser import parse
d = data[0]
clean_data = [(float(d[0].replace('$', '').replace(',', '')), (parse(d[1].split('-')[0].strip()) - datetime.datetime(2017,3,13,0,0)).days, reduce(lambda x,y: y-x, [parse(x.strip()) for x in d[1].split('-')]).days) for d in data]

df = pd.DataFrame(clean_data, columns=['Price', 'Start_Date', 'Trip_Length'])

import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

# Pandas has a ton of built-in visualizations
# Play and Learn
# http://pandas.pydata.org/pandas-docs/stable/visualization.html
df.plot.scatter(x='Start_Date', y='Price')

df['Price'].plot.box()

color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
df['Price'].plot.box(color=color, sym='r+')

df = df.set_value(49, 'Price', 55)
# Time for a Google Investigation
# "IQR Outlier"

# Check out the gallery: 
import seaborn as sns
# this can break matplotlib for some reason...
g = sns.jointplot(df['Start_Date'], df['Price'], kind="kde", size=7, space=0)

import seaborn as sns
# don't blindly set parameters, please read and understand what they mean and how it works
# http://seaborn.pydata.org/tutorial/distributions.html
# lots of great tutorials: http://seaborn.pydata.org/tutorial.html
g = sns.jointplot(df['Start_Date'], df['Price'], kind="kde", size=7, space=0, bw=100)

import matplotlib.pyplot as plt

g = sns.jointplot(x="Start_Date", y="Price", data=df, kind="kde", color="MediumTurquoise")
# https://en.wikipedia.org/wiki/Web_colors
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$X$", "$Y$");

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

# All of pandas' viz is built on top of matplotlib as you might have noticed
# You can get started learning matplotlib here: http://matplotlib.org/users/pyplot_tutorial.html


# df = df.set_value(49, 'Price', 255)
X = StandardScaler().fit_transform(df[['Start_Date', 'Price']])
db = DBSCAN(eps=.5, min_samples=3).fit(X)

labels = db.labels_
clusters = len(set(labels))
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
 
plt.subplots(figsize=(12,8))
 
for k, c in zip(unique_labels, colors):
    class_member_mask = (labels == k)
    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=c,
            markeredgecolor='k', markersize=14)
 
plt.title("Total Clusters: {}".format(clusters), fontsize=14, y=1.01)
df['dbscan_labels'] = db.labels_

df.head()

df.dbscan_labels.unique()

t = X[df.dbscan_labels == 1,:]
t.mean(axis=0)

df

from pandas.tools.plotting import parallel_coordinates
df2 = df[['Trip_Length','Start_Date', 'Price', 'dbscan_labels']]

scaled = StandardScaler().fit_transform(df2[df2.columns[:-1]])
df2 = pd.DataFrame(scaled, columns=df2.columns[:-1])
df2['dbscan_labels'] = df.dbscan_labels

parallel_coordinates(df2, 'dbscan_labels')

for result in results:
    bars = result.find_elements_by_class_name('LJTSM3-w-x')
    
    for bar in bars:
        ActionChains(driver).move_to_element(bar).perform()
        time.sleep(0.0001)
        print (result.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[0].text,
               result.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[1].text)



