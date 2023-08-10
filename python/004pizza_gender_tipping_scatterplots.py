import pandas as pd
import matplotlib.pyplot as pl
get_ipython().magic('matplotlib inline')

data = pd.read_csv(r'C:\Users\angel\OneDrive\Documents\data_training\data\RawDelData.csv')
ylabel = 'Order Amount'
xlabel = 'Tip in $'

#Creating a function so I don't have to duplicate code
def create_scatter(df, title_name):
    pl.scatter(df.Tip, df.OrderAmount)
    n = len(df)
    pl.title('n = %s' %n, fontsize=22)
    pl.suptitle(title_name, fontsize=22)
    pl.ylabel(ylabel, fontsize=18)
    pl.xlabel(xlabel, fontsize=18)
    pl.xlim(-0.25, 15) #The right bound excludes outliers in the visualization but not the sample count
    pl.ylim(-0.30, 80) #DITTO the comment above
    pl.rcParams['figure.figsize'] = (15, 10)
    pl.show()

create_scatter(data, 'All Deliveries')

angel_tip = data.loc[data['PersonWhoDelivered']=='Angel']
create_scatter(angel_tip, 'All of Angel Deliveries')

sam_tip = data.loc[data['PersonWhoDelivered']=='Sammie']
create_scatter(sam_tip, 'All of Sam Deliveries')

angel_female = angel_tip.loc[angel_tip['GenderOfTipper'] == 'Female']
create_scatter(angel_female, 'Angel Deliveries to Females')

angel_male = angel_tip.loc[angel_tip['GenderOfTipper'] == 'Male']
create_scatter(angel_male, 'Angel Deliveries to Males')

#There is an excluded outlier in this data. A $25 tip from a very drunk man. I excluded it from the
#scatter plot so that all of the x axes could be the same

sam_female = sam_tip.loc[sam_tip['GenderOfTipper'] == 'Female']
create_scatter(sam_female, 'Sam Deliveries to Females')

sam_male = sam_tip.loc[sam_tip['GenderOfTipper'] == 'Male']
create_scatter(sam_male, 'Sam Deliveries to Males')

pl.ylabel(ylabel)
pl.xlabel(xlabel)
pl.scatter(angel_male.Tip, angel_male.OrderAmount, color='blue', marker='.', label='Angel--Male')
pl.scatter(angel_female.Tip, angel_female.OrderAmount, color='red', marker='.', label='Angel--Female')
pl.scatter(sam_female.Tip, sam_female.OrderAmount, color='black', marker='.', label='Sam--Male')
pl.scatter(sam_male.Tip, sam_male.OrderAmount, color='green', marker='.', label='Sam--Female')
n = len(angel_male) + len(angel_female) + len(sam_female) + len(sam_male)
pl.title('n = %s' %n)
pl.title('n = %s' %n, fontsize=22)
pl.suptitle('All Deliveries - Again', fontsize =22)    
pl.ylabel(ylabel, fontsize=18)
pl.xlabel(xlabel, fontsize=18)
pl.xlim(-0.25, 15)
pl.ylim(-0.30, 80)
pl.rcParams['figure.figsize'] = (20, 15)
pl.legend()
pl.show()

import numpy as np

f, axarr = pl.subplots(2, 2)
xmin = -0.25
xmax = 15
ymin = -0.30
ymax = 80


axarr[0, 0].set_xlim([xmin, xmax]) #I tried a for each loop so I didn't have to repeat this code 4 times but couldn't
axarr[0, 0].set_ylim([ymin, ymax]) #figure it out. something like  #for i in axarr[0, :]:
axarr[0, 0].scatter(angel_male.Tip, angel_male.OrderAmount)            #axarr[i, :].set_xlim()
axarr[0, 0].set_title('Angel Male Deliveries')

axarr[0, 1].set_xlim([xmin, xmax])
axarr[0, 1].set_ylim([ymin, ymax]) 
axarr[0, 1].scatter(sam_male.Tip, sam_male.OrderAmount)
axarr[0, 1].set_title('Sam Male Deliveries')

axarr[1, 0].set_xlim([xmin, xmax])
axarr[1, 0].set_ylim([ymin, ymax]) 
axarr[1, 0].scatter(angel_female.Tip, angel_female.OrderAmount)
axarr[1, 0].set_title('Angel Female Deliveries')

axarr[1, 1].set_xlim([xmin, xmax])
axarr[1, 1].set_ylim([ymin, ymax]) 
axarr[1, 1].scatter(sam_female.Tip, sam_female.OrderAmount)
axarr[1, 1].set_title('Sam Female Deliveries')



