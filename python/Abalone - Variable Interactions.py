get_ipython().magic('matplotlib notebook')

# numbers
import numpy as np
import pandas as pd

# stats
import statsmodels.api as sm
import scipy.stats as stats

# plots
import matplotlib.pyplot as plt
import seaborn as sns

# utils
import os, re

# Load data, filter outliers (zero height/length)

def abalone_load(data_file, infant=False):
    # x data labels
    xnlabs = ['Sex']
    xqlabs = ['Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight']
    xlabs = xnlabs + xqlabs

    # y data labels
    ylabs = ['Rings']

    # data
    df = pd.read_csv(data_file, header=None, sep=' ', names=xlabs+ylabs)
    
    if(infant):
        new_df = df[ df['Sex']=='I' ]
    else:
        new_df = df[ df['Sex']<>'I' ]
    
    return new_df

def adult_abalone_load(data_file):
    return abalone_load(data_file,False)

def abalone_removeoutliers(df,verbose=False):
    len1 = len(df)
    df = df[ df['Height'] < 0.30 ]
    df = df[ df['Height'] > 0.001 ]
    df = df[ df['Length'] > 0.001 ]
    df = df[ df['Diameter'] > 0.001 ]
    len2 = len(df)
    if(verbose):
        print "Removed",(len1-len2),"outliers"
    return df

# x data labels
xnlabs = ['Sex']
xqlabs = ['Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight']
xlabs = xnlabs + xqlabs

# y data labels
ylabs = ['Rings']

adt_df = abalone_removeoutliers(adult_abalone_load('abalone/Dataset.data'),True)

print adt_df.columns

adt_df['Length (Normalized)']    = adt_df['Length']/adt_df['Length'].mean()
adt_df['Diameter (Normalized)']  = adt_df['Diameter']/adt_df['Diameter'].mean()
adt_df['Height (Normalized)']    = adt_df['Height']/adt_df['Height'].mean()

adt_df['Length (Max Normalized)']    = adt_df['Length']/adt_df['Length'].max()
adt_df['Diameter (Max Normalized)']  = adt_df['Diameter']/adt_df['Diameter'].max()
adt_df['Height (Max Normalized)']    = adt_df['Height']/adt_df['Height'].max()

adt_df['Volume']                 = adt_df['Length']*adt_df['Diameter']*adt_df['Height']


adt_df['Volume (Normalized)']    = adt_df['Length (Normalized)']*adt_df['Diameter (Normalized)']*adt_df['Height (Normalized)']
adt_df['Dimension (Normalized)'] = adt_df['Volume (Normalized)'].apply( lambda x : pow(x,1.0/3.0) )

adt_df['Volume (Max Normalized)']    = adt_df['Length (Max Normalized)']*adt_df['Diameter (Max Normalized)']*adt_df['Height (Max Normalized)']
adt_df['Volume (VMax Normalized)'] = adt_df['Volume']/adt_df['Volume'].max()




adt_df['Volume (Log Normalized)'] = adt_df['Volume (Normalized)'].apply( lambda x : np.log(x) )
adt_df['Volume (Log Max Normalized)'] = adt_df['Volume (Max Normalized)'].apply( lambda x : np.log(x) )
adt_df['Volume (Log)']            = adt_df['Volume'].apply( lambda x : np.log(x) )



adt_df['Area']                    = 3.14159*adt_df['Diameter'].apply(lambda x : x*x)
adt_df['Inverse Volume']          = adt_df['Volume'].apply( lambda x : x / (1.00000 - x) )

adt_df['Viscera-Shell Weight Ratio']            = adt_df['Viscera weight']/adt_df['Shell weight']
adt_df['Viscera-Shucked Weight Ratio']            = adt_df['Viscera weight']/adt_df['Shucked weight']
adt_df['Viscera-Whole Weight Ratio']            = adt_df['Viscera weight']/adt_df['Whole weight']

adt_df['Whole weight (Normalized)']   = adt_df['Whole weight']/adt_df['Whole weight'].max()
adt_df['Shell weight (Normalized)']   = adt_df['Shell weight']/adt_df['Shell weight'].max()
adt_df['Shucked weight (Normalized)'] = adt_df['Shucked weight']/adt_df['Shucked weight'].max()
adt_df['Viscera weight (Normalized)'] = adt_df['Viscera weight']/adt_df['Viscera weight'].max()

adt_df['Rings (Log)']             = adt_df['Rings'].apply( lambda x : np.log(x) )
adt_df['Rings (Normalized)']      = adt_df['Rings'].apply( lambda x : float(x)  )/adt_df['Rings'].max()

fig = plt.figure()
sns.distplot( adt_df['Volume (Log Normalized)'], label="Volume (Log Norm)")
sns.distplot( adt_df['Volume (Normalized)'], label="Volume (Normalized)")
plt.legend()

fig = plt.figure()

labels = ['Viscera-Shell Weight Ratio','Viscera-Shucked Weight Ratio','Viscera-Whole Weight Ratio']
for lab in labels:
    sns.distplot( adt_df[lab], label = lab)

plt.legend()
plt.show()

fig = plt.figure()

plt.plot( adt_df['Length (Normalized)'],   adt_df['Rings'], 'r*' )
plt.plot( adt_df['Diameter (Normalized)'], adt_df['Rings'], 'b*' )
plt.plot( adt_df['Height (Normalized)'],   adt_df['Rings'], 'g*' )

plt.show()

plt.figure()

#colors=adt_df['Rings'].apply(lambda x : float(x)).values
colors=adt_df['Rings (Log)']
colors -= colors.min()
colors *= (1.0/colors.max())
cm = plt.cm.get_cmap('RdYlBu')
cm = plt.cm.ocean
cm = plt.cm.copper
cm = plt.cm.magma
cm = plt.cm.RdBu
cm = plt.cm.Purples
cm = plt.cm.OrRd

cm = plt.cm.jet
plot_color = cm(colors)

#plt.plot( adt_df['Volume (Log Normalized)'], adt_df['Rings'], 'b*' )
plt.scatter( adt_df['Volume (Log Normalized)'], adt_df['Viscera-Shell Weight Ratio'], marker='*', alpha=0.3, c=plot_color)
plt.xlabel("Volume (Log Normalized)")
plt.ylabel("Viscera-Shell Weight Ratio")
plt.show()

#colors=adt_df['Rings'].apply(lambda x : float(x)).values
colors=adt_df['Rings (Log)']
colors -= colors.min()
colors *= (1.0/colors.max())
cm = plt.cm.get_cmap('RdYlBu')
cm = plt.cm.jet

plot_color = cm(colors)


fig = plt.figure(figsize=(14,8))
ax1,ax2,ax3 = fig.add_subplot(231), fig.add_subplot(232), fig.add_subplot(233)
ax4,ax5,ax6 = fig.add_subplot(234), fig.add_subplot(235), fig.add_subplot(236)


sct1 = ax1.scatter(adt_df['Volume (Log Normalized)'], adt_df['Viscera-Shell Weight Ratio'] , marker='*', c=plot_color, alpha=0.5)
ax1.set_xlabel('Volume (Log Normalized)')
ax1.set_ylabel('Viscera-Shell Weight Ratio')

sct2 = ax2.scatter(adt_df['Volume (Log Normalized)'], adt_df['Viscera-Shucked Weight Ratio'] , marker='*', c=plot_color, alpha=0.5)
ax2.set_xlabel('Volume (Log Normalized)')
ax2.set_ylabel('Viscera-Shucked Weight Ratio')

sct3 = ax3.scatter(adt_df['Volume (Log Normalized)'], adt_df['Viscera-Whole Weight Ratio'] , marker='*', c=plot_color, alpha=0.5)
ax3.set_xlabel('Volume (Log Normalized)')
ax3.set_ylabel('Viscera-Whole Weight Ratio')


sct4 = ax4.scatter(adt_df['Volume (Normalized)'], adt_df['Viscera-Shell Weight Ratio'] , marker='*', c=plot_color, alpha=0.5)
ax4.set_xlabel('Volume (Normalized)')
ax4.set_ylabel('Viscera-Shell Weight Ratio')

sct5 = ax5.scatter(adt_df['Volume (Normalized)'], adt_df['Viscera-Shucked Weight Ratio'] , marker='*', c=plot_color, alpha=0.5)
ax5.set_xlabel('Volume (Normalized)')
ax5.set_ylabel('Viscera-Shucked Weight Ratio')

sct6 = ax6.scatter(adt_df['Volume (Normalized)'], adt_df['Viscera-Whole Weight Ratio'] , marker='*', c=plot_color, alpha=0.5)
ax6.set_xlabel('Volume (Normalized)')
ax6.set_ylabel('Viscera-Whole Weight Ratio')


plt.show()

colors=adt_df['Rings'].apply(lambda x : float(x)).values

#colors=adt_df['Rings (Log)'].values

colors -= colors.min()
colors *= (1.0/colors.max())

cm = plt.cm.jet
plot_color = cm(colors)

fig = plt.figure(figsize=(14,10))
ax1,ax2,ax3 = fig.add_subplot(231), fig.add_subplot(232), fig.add_subplot(233)
ax4,ax5,ax6 = fig.add_subplot(234), fig.add_subplot(235), fig.add_subplot(236)


sc1 = ax1.scatter(adt_df['Volume (Log Normalized)'], adt_df['Whole weight (Normalized)'] , marker='*', c=plot_color, alpha=0.3)
ax1.set_xlabel('Volume (Log Normalized)')
ax1.set_ylabel('Whole weight (Normalized)')

sc2 = ax2.scatter(adt_df['Volume (Normalized)'], adt_df['Whole weight (Normalized)'] , marker='*', c=plot_color, alpha=0.3)
ax2.set_xlabel('Volume (Normalized)')
ax2.set_ylabel('Whole weight (Normalized)')

sc3 = ax3.scatter(adt_df['Volume (Log Max Normalized)'], adt_df['Whole weight (Normalized)'] , marker='*', c=plot_color, alpha=0.3)
ax3.set_xlabel('Volume (Log Max Normalized)')
ax3.set_ylabel('Whole weight (Normalized)')



sc4 = ax4.scatter(adt_df['Volume (Log Normalized)'], adt_df['Shucked weight (Normalized)'] , marker='*', c=plot_color, alpha=0.3)
ax4.set_xlabel('Volume (Log Normalized)')
ax4.set_ylabel('Shucked weight (Normalized)')

sc5 = ax5.scatter(adt_df['Volume (Normalized)'], adt_df['Shucked weight (Normalized)'] , marker='*', c=plot_color, alpha=0.3)
ax5.set_xlabel('Volume (Normalized)')
ax5.set_ylabel('Shucked weight (Normalized)')

sc6 = ax6.scatter(adt_df['Volume (Log Max Normalized)'], adt_df['Shucked weight (Normalized)'] , marker='*', c=plot_color, alpha=0.3)
ax6.set_xlabel('Volume (Log Max Normalized)')
ax6.set_ylabel('Shucked weight (Normalized)')




plt.show()

colors=adt_df['Rings'].apply(lambda x : float(x)).values
#colors=adt_df['Rings (Log)'].values

colors -= colors.min()
colors *= (1.0/colors.max())

cm = plt.cm.jet
plot_color = cm(colors)

fig = plt.figure(figsize=(14,10))
ax1,ax2,ax3 = fig.add_subplot(231), fig.add_subplot(232), fig.add_subplot(233)
ax4,ax5,ax6 = fig.add_subplot(234), fig.add_subplot(235), fig.add_subplot(236)


sc1 = ax1.scatter(adt_df['Volume (Log Normalized)'], adt_df['Shell weight (Normalized)'] , marker='*', c=plot_color, alpha=0.3)
ax1.set_xlabel('Volume (Log Normalized)')
ax1.set_ylabel('Shell weight (Normalized)')

sc2 = ax2.scatter(adt_df['Volume (Normalized)'], adt_df['Shell weight (Normalized)'] , marker='*', c=plot_color, alpha=0.3)
ax2.set_xlabel('Volume (Normalized)')
ax2.set_ylabel('Shell weight (Normalized)')

sc3 = ax3.scatter(adt_df['Volume (Log Max Normalized)'], adt_df['Shell weight (Normalized)'] , marker='*', c=plot_color, alpha=0.3)
ax3.set_xlabel('Volume (Log Max Normalized)')
ax3.set_ylabel('Shell weight (Normalized)')



sc4 = ax4.scatter(adt_df['Volume (Log Normalized)'], adt_df['Viscera weight (Normalized)'] , marker='*', c=plot_color, alpha=0.3)
ax4.set_xlabel('Volume (Log Normalized)')
ax4.set_ylabel('Viscera weight (Normalized)')

sc5 = ax5.scatter(adt_df['Volume (Normalized)'], adt_df['Viscera weight (Normalized)'] , marker='*', c=plot_color, alpha=0.3)
ax5.set_xlabel('Volume (Normalized)')
ax5.set_ylabel('Viscera weight (Normalized)')

sc6 = ax6.scatter(adt_df['Volume (Log Max Normalized)'], adt_df['Viscera weight (Normalized)'] , marker='*', c=plot_color, alpha=0.3)
ax6.set_xlabel('Volume (Log Max Normalized)')
ax6.set_ylabel('Viscera weight (Normalized)')

plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,4))

colors=adt_df['Rings (Log)'].values
colors -= colors.min()
colors *= (1.0/colors.max())
cm = plt.cm.get_cmap('RdYlBu')
cm = plt.cm.jet
plot_color = cm(colors)


ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')


ax1.scatter(adt_df['Volume (Normalized)'], adt_df['Whole weight (Normalized)'] , adt_df['Rings (Log)'], c=plot_color)
ax2.scatter(adt_df['Volume (Log Normalized)'], adt_df['Whole weight (Normalized)'] , adt_df['Rings (Log)'], c=plot_color)

ax1.set_xlabel('Volume (Normalized)')
ax1.set_ylabel('Whole weight (Normalized)')
ax1.set_zlabel('Rings (Log)')

ax2.set_xlabel('Volume (Log Normalized)')
ax2.set_ylabel('Whole weight (Normalized)')
ax2.set_zlabel('Rings (Log)')

plt.show()

fig = plt.figure(figsize=(10,4))

colors=adt_df['Rings'].apply(lambda x : float(x) ).values
colors -= colors.min()
colors *= (1.0/colors.max())
cm = plt.cm.get_cmap('RdYlBu')
cm = plt.cm.jet
plot_color = cm(colors)


ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')


ax1.scatter(adt_df['Volume (Normalized)'], adt_df['Whole weight (Normalized)'] , adt_df['Rings'], c=plot_color)
ax2.scatter(adt_df['Volume (Log Normalized)'], adt_df['Whole weight (Normalized)'] , adt_df['Rings'], c=plot_color)

ax1.set_xlabel('Volume (Normalized)')
ax1.set_ylabel('Whole weight (Normalized)')
ax1.set_zlabel('Rings')

ax2.set_xlabel('Volume (Log Normalized)')
ax2.set_ylabel('Whole weight (Normalized)')
ax2.set_zlabel('Rings')

plt.show()

adt_df['Growth Magnitude'] = ( adt_df['Rings (Normalized)'].apply(lambda x:x*x)                          + adt_df['Volume (Normalized)'].apply(lambda x:x*x)                         + adt_df['Whole weight (Normalized)'].apply(lambda x:x*x) )
adt_df['Growth Magnitude'] = adt_df['Growth Magnitude'].apply(lambda x : np.sqrt(x))

fig = plt.figure()
sns.distplot(adt_df['Growth Magnitude'])
plt.show()

adt_df['Growth Rate'] = ( adt_df['Volume (Normalized)'].apply(lambda x:x*x)                         + adt_df['Whole weight (Normalized)'].apply(lambda x:x*x) )
adt_df['Growth Rate'] = adt_df['Growth Rate'].apply(lambda x : np.sqrt(x))
adt_df['Growth Rate'] = adt_df['Growth Rate']/adt_df['Rings (Normalized)']

fig = plt.figure()
sns.distplot(adt_df['Growth Rate'])
plt.show()

adt_df['SexNum'] = adt_df['Sex'].eq('M').map(lambda x : float(x))

fig = plt.figure()
sns.distplot(adt_df['Growth Rate'][adt_df['SexNum']>0.9],label='male')
sns.distplot(adt_df['Growth Rate'][adt_df['SexNum']<0.1],label='female')
plt.legend()
plt.show()

colors = adt_df['Growth Rate']
colors -= colors.min()
colors *= (1.0/colors.max())
cm = plt.cm.get_cmap('RdYlBu')
cm = plt.cm.jet
plot_color = cm(colors)

fig = plt.figure()

plt.scatter( adt_df['Diameter (Normalized)'], adt_df['Height (Normalized)'] , alpha=0.3, marker='*', c=plot_color)
plt.show()

fig = plt.figure(figsize=(6,4))


colors=adt_df['SexNum']
colors -= colors.min()
colors *= (1.0/colors.max())
cm = plt.cm.copper
plot_color = cm(colors)

ax1 = fig.add_subplot(111, projection='3d')

ax1.scatter(adt_df['Volume (Max Normalized)'], adt_df['Whole weight (Normalized)'] , adt_df['Rings'], c=plot_color)

ax1.set_xlabel('Volume (Max Normalized)')
ax1.set_ylabel('Whole weight (Normalized)')
ax1.set_zlabel('Num Rings')

plt.show()





