get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import pandas as pd

# 1 row and 2 column figure
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,7))
# when subplot has more than one row or column, the second argument
# returned by `plt.subplots` is a tuple.

# two axes objects
ax1 = axes[0]
ax2 = axes[1]

# plot on first axes object (aka first subplot at posn row=1, column=1)
ax1.plot(np.arange(10), np.random.randint(1, 99, size=10), label='Philadelphia')
ax1.plot(np.arange(10), np.random.randint(1, 99, size=10), label='Boston')
ax1.set_title('A tale of two cities')
ax1.grid(which='major')
ax1.legend()

# plot on second axes object (aka subplot at posn row=1, column=2)
theta = np.linspace(-np.pi, np.pi, 100)
ax2.plot(theta, np.sin(theta), ls='-.', label='Sine')
ax2.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax2.set_xticklabels(['$-\pi$', '$-\pi/2$', 0, '$\pi/2$', '$\pi$'])
ax2.set_title("Sine Wave")
ax2.grid(which='major')
ax2.legend()

# Add a centered title for the whole figure
fig.suptitle('My Fancy Subplot')
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(10, 5))
plt.subplots_adjust(wspace=0.5, hspace=0.3,
                    left=0.125, right=0.9,
                    top=0.9,    bottom=0.1)

def example_plot(ax):
    ax.plot(np.random.randn(100), color=np.random.rand(3,1))
    ax.set_xlabel('X points', fontsize=10)
    ax.set_ylabel('Y points', fontsize=10)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,5))
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)
# try commenting the below line and see the difference
plt.tight_layout()

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
theta = np.linspace(-np.pi, np.pi, 100)

# Sine plot
ax1.plot(theta, np.sin(theta), ls='-.', label='Sine', color='darkorange')
ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax1.set_xticklabels(['$-\pi$', '$-\pi/2$', 0, '$\pi/2$', '$\pi$'])
ax1.legend()

ax1.set_xlim(-np.pi, np.pi)

# Cosine plot
ax2.plot(theta, np.cos(theta), ls='-.', label='Cosine', color='darkolivegreen')
ax2.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax2.set_xticklabels(['$-\pi$', '$-\pi/2$', 0, '$\pi/2$', '$\pi$'])
ax2.legend()

fig = plt.figure(figsize=(15, 7))
grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)
ax0 = plt.subplot(grid[0, 0])
ax1 = plt.subplot(grid[0, 1:])
ax2 = plt.subplot(grid[1, :2])
ax3 = plt.subplot(grid[1, 2])

theta = np.linspace(-np.pi, np.pi, 100)

ax0.plot(theta, np.sin(theta), label='Sine', color='#1abc9c')
ax1.plot(theta, np.tan(theta), label='Tan', color='#3498db')
ax2.plot(theta, np.cos(theta), label='Cosine', color='#e74c3c')
ax3.plot(theta, np.log(theta), label='log', color='#9b59b6')

ax0.legend()
ax1.legend()
ax2.legend()
ax3.legend()

noise = pd.read_csv('sample_datasets/2012_NYC_Noise_Complaints.csv', 
                    parse_dates=['Created Date'])

import matplotlib as mpl
# All the complaints aggregated by day
total_complaints = noise.groupby(by=noise['Created Date'])['Unique Key'].count()
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(total_complaints, color='#3498db', label='total complaints')
ax.legend()

fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(total_complaints, color='#3498db', label='total complaints')

# formatting the xaxis
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=1))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter("%B'%y"));
ax.legend()

# Let point out some occassions
fav_dates={"2016-11-23":"Thanks Giving", 
          "2016-12-25": "Christmas",
          "2016-12-31": "New Years",
          "2017-1-1":   "First Day after New Year"
          }
for date in fav_dates.items():
    ax.text(date[0], total_complaints.loc[date[0]], date[1])

fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(total_complaints, color='#3498db', label='total complaints')

# formatting the xaxis
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=1))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter("%B'%y"));
ax.legend()

# Let point out some occassions
fav_dates={"2016-11-23":"Thanks Giving", 
          "2016-12-25": "Christmas",
          "2016-12-31": "New Years",
          "2017-1-1":   "First Day after New Year"
          }
for date in fav_dates.items():
    xposition = date[0]
    yposition = total_complaints.loc[date[0]]
    ax.annotate(date[1], xy=(xposition, yposition),
               xycoords='data', ha='center',
               xytext=(0, 20), textcoords='offset points',
               arrowprops=dict(arrowstyle="->"))

fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(total_complaints, color='#3498db', label='total complaints')

# formatting the xaxis
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=1))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter("%B'%y"));
ax.legend()

# Let point out some occassions
fav_dates={"2016-11-23":"Thanks Giving", 
          "2016-12-25": "Christmas",
          "2016-12-31": "New Years",
          "2017-1-1":   "First Day after New Year"
          }
for date in fav_dates.items():
    xposition = date[0]
    yposition = total_complaints.loc[date[0]]
    ax.annotate(date[1], xy=(xposition, yposition),
               xycoords='data', ha='center',
               xytext=(50, 25), textcoords='offset points',
               arrowprops=dict(arrowstyle="->",
                               connectionstyle="arc3,rad=0.5",))

# For sake of completion
ax.set_title("NYC 311 Noise Complaints (Nov'16 to Jan'17)", 
             fontdict=dict(fontsize=14, 
                           fontweight='bold'))
ax.set_ylabel('Number of Complaints')

