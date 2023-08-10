get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn  # make charts prettier and more readable

# Let's look at the number of sessions first
sessions = pd.read_csv(
    'sessions.csv',
    index_col='Day Index',
    parse_dates=True
)

def intify_sessions(df):
    """
    GA returns big numbers as string like 1,000
    That won't do here we need ints
    """
    return df['Sessions'].replace(',', '', regex=True).astype('int')

# We need to ensure we only have ints
sessions['Sessions'] = intify_sessions(sessions)

# We will annotate the data with our articles to see the 
# difference between them
# The number at the end of the tuples is the y offset 
# for the annotation
articles = [
    ("2015-05-11", "Berg's Little Printer", 30),
    ("2015-05-19", "Comparing the weather of places I've lived in", 130),
    ("2015-05-25", "My experience of using NixOps as an Ansible user", 135),
    ("2015-06-05", "Using Protobuf instead of JSON to communicate with a frontend", 10),
]
dates = [date.isoformat() for date in sessions.index.date.tolist()]

plt.figure(figsize=(16, 10))
ax = plt.subplot(111)
plt.tick_params(axis='both', which='major', labelsize=16)
sessions['Sessions'].plot(x_compat=True)

plt.title("Daily sessions on the blog", fontsize=18)

for (date, title, offset) in articles:
    index = dates.index(date)
    number_sessions = sessions['Sessions'][index]
    _ = ax.annotate(
        '%s: %d sessions' % (title, number_sessions),
        xy=(sessions.index[index], sessions['Sessions'][index]),
        fontsize= 20,
        horizontalalignment='right', verticalalignment='center',
        xytext=(00, offset),
        textcoords='offset points',
        arrowprops=dict(arrowstyle='-|>'))

channels = pd.read_csv('channels.csv')
# We need to ensure we only have ints
channels['Sessions'] = intify_sessions(channels)
plt.figure()
wedges = channels['Sessions'].plot(
    kind='pie',
    autopct='%1.1f%%',
    labels=channels['Default Channel Grouping'],
    fontsize=16,
    labeldistance=1.1
)

_ = plt.axis('equal')

locations = pd.read_csv('locations.csv')
locations['Sessions'] = intify_sessions(locations)

plt.figure(figsize=(8,6))
ax = locations['Sessions'].plot(kind='barh')
ax.set_yticklabels(locations['Country'])
ax.invert_yaxis()
plt.tick_params(axis='both', which='major', labelsize=16)
plt.title("Number of sessions by countries (top 10)")
_ = ax.set_xlabel("Number of sessions")

oses = pd.read_csv('os.csv')
oses['Sessions'] = intify_sessions(oses)
plt.figure()
# Limiting to OSes over 200 users, sorry WP, Chrome OS, 
# Firefox OS and Free/OpenBSD users
oses[oses['Sessions'] > 200]['Sessions'].plot(
    kind='pie',
    autopct='%1.1f%%',
    labels=oses['Operating System'],
    fontsize=16,
    labeldistance=1.1
)
_ = plt.axis('equal')

browsers = pd.read_csv('browsers.csv')
browsers['Sessions'] = intify_sessions(browsers)
# Limiting to browsers over 200 users, 
# sorry IE/Opera/Blackberry users
browsers[browsers['Sessions'] > 200]['Sessions'].plot(
    kind='pie',
    autopct='%1.1f%%',
    labels=browsers['Browser'],
    fontsize=16
)
_ = plt.axis('equal')

