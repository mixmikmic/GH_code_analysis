import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns

get_ipython().magic('matplotlib inline')

dframe = pd.read_csv('data/innermelbourne.csv')
dframe.columns = dframe.columns.str.strip() #To fix formatting issues with headers

#Dropping columns deemed irrelevant
aframe = dframe.drop(['gcc_name11','gcc_code11','sa2_5dig11','sa1_7dig11','sa3_code11','sa4_code11','ste_code11','ste_name11'],axis=1)

#Group by SA2 suburb
avg_sa2 = aframe[['sa2_name11','SumZScore']].groupby('sa2_name11').mean()

#Group by SA3 area
avg_sa3 = aframe[['sa3_name11','SumZScore']].groupby('sa3_name11').mean()

avg_sa3

avg_sa2

sframe = pd.read_csv('data/sydcityinnersouthresults.csv')
sframe.columns = sframe.columns.str.strip()

bframe = sframe.drop(['gcc_name11','gcc_code11','sa2_5dig11','sa1_7dig11','sa3_code11','sa4_code11','ste_code11','ste_name11'],axis=1)

#Counting the SA1s in Melbourne and Sydney
print "SA1s in Melbourne: "+ str(len(aframe.index))
print "SA1s in Sydney: "+ str(len(bframe.index))

#Group by SA2 suburb
savg_sa2 = bframe[['sa2_name11','SumZScore']].groupby('sa2_name11').mean()
# savg_sa2.reset_index()
# savg_sa2.columns = ['SA2 Name','Walkability Index']

#Group by SA3 area
savg_sa3 = bframe[['sa3_name11','SumZScore']].groupby('sa3_name11').mean()
# savg_sa2.reset_index()
# savg_sa3.columns = ['SA2 Name','Walkability Index']

#Counting the SA1s in Melbourne and Sydney
print "SA1s in Melbourne: "+ str(len(avg_sa2.index))
print "SA1s in Sydney: "+ str(len(savg_sa2.index))

savg_sa3

savg_sa2

#Top 5 SA2 suburbs Melbourne
top_sa2_melb = avg_sa2.sort_values('SumZScore',ascending=False).head(5).plot(kind='bar',figsize=(8,6))

top_sa2_melb.set_ylabel('Walkability Index') #Setting the Y label
top_sa2_melb.set_xlabel('SA2 Suburb') #Setting the X label
top_sa2_melb.legend(['Walkability Index'])
fig_top_sa2_melb = top_sa2_melb.get_figure() #Assigning the figure to a new variable for further operations
fig_top_sa2_melb.tight_layout() #To fit everything in the saved image
fig_top_sa2_melb.subplots_adjust(top=0.93)
fig_top_sa2_melb.suptitle('Top 5 Melbourne SA2 Suburbs based on Walkability',fontsize=15,fontweight='bold') #Title for the figure
fig_top_sa2_melb.savefig("figures/top_5_melbourne_sa2.png") 

#Top 5 Suburbs Sydney
top_sa2_syd = savg_sa2.sort_values('SumZScore',ascending=False).head(5).plot(kind='bar',figsize=(8,8))


top_sa2_syd.set_ylabel('Walkability Index') #Setting the Y label
top_sa2_syd.set_xlabel('SA2 Suburb') #Setting the X label
top_sa2_syd.legend(['Walkability Index'])
fig_top_sa2_syd = top_sa2_syd.get_figure() #Assigning the figure to a new variable for further operations
fig_top_sa2_syd.tight_layout() #To fit everything in the saved image
fig_top_sa2_syd.subplots_adjust(top=0.93)
fig_top_sa2_syd.suptitle('Top 5 Sydney SA2 Suburbs based on Walkability',fontsize=15,fontweight='bold') #Title for the figure
fig_top_sa2_syd.savefig("figures/top_5_sydney_sa2.png") 

#Least 5 Suburbs Melbourne
bot_sa2_melb = avg_sa2.sort_values('SumZScore',ascending=True).head(5).plot(kind='bar',figsize=(8,6))

bot_sa2_melb.set_ylabel('Walkability Index') #Setting the Y label
bot_sa2_melb.set_xlabel('SA2 Suburb') #Setting the X label
bot_sa2_melb.legend(['Walkability Index'],loc=4)
fig_bot_sa2_melb = bot_sa2_melb.get_figure() #Assigning the figure to a new variable for further operations
fig_bot_sa2_melb.tight_layout() #To fit everything in the saved image
fig_bot_sa2_melb.subplots_adjust(top=0.93)
fig_bot_sa2_melb.suptitle('Bottom 5 Melbourne SA2 Suburbs based on Walkability',fontsize=15,fontweight='bold') #Title for the figure
fig_bot_sa2_melb.savefig("figures/bottom_5_melbourne_sa2.png")

bot_sa2_syd = savg_sa2.sort_values('SumZScore').head(5).plot(kind='bar',figsize=(8,6))

bot_sa2_syd.set_ylabel('Walkability Index') #Setting the Y label
bot_sa2_syd.set_xlabel('SA2 Suburb') #Setting the X label
bot_sa2_syd.legend(['Walkability Index'],loc=4)
fig_bot_sa2_syd = bot_sa2_syd.get_figure() #Assigning the figure to a new variable for further operations
fig_bot_sa2_syd.tight_layout() #To fit everything in the saved image
fig_bot_sa2_syd.subplots_adjust(top=0.93)
fig_bot_sa2_syd.suptitle('Bottom 5 Sydney SA2 Suburbs based on Walkability',fontsize=15,fontweight='bold') #Title for the figure
fig_bot_sa2_syd.savefig("figures/bottom_5_sydney_sa2.png")

#Calculating the average walkabiity score for Inner Melbourne
avg_sa4 = aframe[['sa4_name11','SumZScore']].round(2).groupby('sa4_name11').mean()
avg_sa4

#Calculating the average walkability score for Sydney City and Inner South
savg_sa4 = bframe[['sa4_name11','SumZScore']].round(2).groupby('sa4_name11').mean()
savg_sa4

melb_syd_avg_sa4_fig = pd.concat([avg_sa4,savg_sa4]).plot(kind='bar',figsize=(6,7))

melb_syd_avg_sa4_fig.set_ylabel('Walkability Index') #Setting the Y label
melb_syd_avg_sa4_fig.set_xlabel('SA4 Region') #Setting the X label
melb_syd_avg_sa4_fig.legend(['Walkability Index'],loc=1)
fig_melb_syd_avg_sa4 = melb_syd_avg_sa4_fig.get_figure() #Assigning the figure to a new variable for further operations
fig_melb_syd_avg_sa4.tight_layout() #To fit everything in the saved image
fig_melb_syd_avg_sa4.subplots_adjust(top=0.93)
fig_melb_syd_avg_sa4.suptitle('Comparison of SA4 Regions - Mean',fontsize=15,fontweight='bold') #Title for the figure
fig_melb_syd_avg_sa4.savefig("figures/melb_syd_avg_sa4.png")

median_sa4 = aframe[['sa4_name11','SumZScore']].round(2).groupby('sa4_name11').median()
median_sa4

smedian_sa4 = bframe[['sa4_name11','SumZScore']].round(2).groupby('sa4_name11').median()
smedian_sa4

melb_syd_median_sa4_fig = pd.concat([median_sa4,smedian_sa4]).plot(kind='bar',figsize=(8,6))

melb_syd_median_sa4_fig.set_ylabel('Walkability Index') #Setting the Y label
melb_syd_median_sa4_fig.set_xlabel('SA4 Region') #Setting the X label
melb_syd_median_sa4_fig.legend(['Walkability Index'],loc=1)
fig_melb_syd_median_sa4 = melb_syd_median_sa4_fig.get_figure() #Assigning the figure to a new variable for further operations
fig_melb_syd_median_sa4.tight_layout() #To fit everything in the saved image
fig_melb_syd_median_sa4.subplots_adjust(top=0.93)
fig_melb_syd_median_sa4.suptitle('Comparison of SA4 Regions - Median',fontsize=15,fontweight='bold') #Title for the figure
fig_melb_syd_median_sa4.savefig("figures/melb_syd_median_sa4.png")

aframe_no_southbank = aframe[aframe.sa2_name11 != 'Southbank']
bframe_no_banksmeadow = bframe[bframe.sa2_name11 != 'Banksmeadow']

avg_sa4_no_south = aframe_no_southbank[['sa4_name11','SumZScore']].round(2).groupby('sa4_name11').mean()
avg_sa4_no_south

savg_sa4_no_banks = bframe_no_banksmeadow[['sa4_name11','SumZScore']].round(2).groupby('sa4_name11').mean()
savg_sa4_no_banks

melb_syd_avg_sa4_no_fig = pd.concat([avg_sa4_no_south,savg_sa4_no_banks]).plot(kind='bar',figsize=(8,6))

melb_syd_avg_sa4_no_fig.set_ylabel('Walkability Index') #Setting the Y label
melb_syd_avg_sa4_no_fig.set_xlabel('SA4 Region') #Setting the X label
melb_syd_avg_sa4_no_fig.legend(['Walkability Index'],loc=1)
fig_melb_syd_avg_sa4_no = melb_syd_avg_sa4_no_fig.get_figure() #Assigning the figure to a new variable for further operations
fig_melb_syd_avg_sa4_no.tight_layout() #To fit everything in the saved image
fig_melb_syd_avg_sa4_no.subplots_adjust(top=0.93)
fig_melb_syd_avg_sa4_no.suptitle('Comparison of SA4 Regions - Mean | Without outliers',fontsize=15,fontweight='bold') #Title for the figure
fig_melb_syd_avg_sa4_no.savefig("figures/melb_syd_avg_sa4_no_outliers.png")

#All means
print avg_sa4 
print savg_sa4
print avg_sa4_no_south
print savg_sa4_no_banks
print "\n\n"
print "Differences"

print abs(avg_sa4 - avg_sa4_no_south)
print abs(savg_sa4 - savg_sa4_no_banks)

