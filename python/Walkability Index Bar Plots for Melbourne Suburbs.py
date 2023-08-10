#We don't use seaborn in this particular notebook, but it makes matplotlib charts look nicer.
import seaborn as sns 

#Pandas for data analysis
import pandas as pd
from pandas import Series,DataFrame

#To display the figures in the notebook itself.
get_ipython().magic('matplotlib inline')

#Importing the walkability data file for Inner Melbourne. 
dframe = pd.read_csv('data/innermelbourne.csv')
dframe.columns = dframe.columns.str.strip() #Stripping white-spaces off of column names. 

#Displaying the Dataframe.
dframe

#Dropping columns deemed irrelevant
aframe = dframe.drop(['gcc_name11','gcc_code11','sa2_5dig11','sa1_7dig11','sa3_code11','sa4_code11','ste_code11','ste_name11'],axis=1)

#Group by SA2 suburb
avg_sa2 = aframe[['sa2_name11','SumZScore']].groupby('sa2_name11').mean().round(3)

#Group by SA3 area
avg_sa3 = aframe[['sa3_name11','SumZScore']].groupby('sa3_name11').mean().round(3)

avg_sa3

#Changing the column name from SumZScore to Walkability Index, for better readability.
avg_sa3.columns=['Walkability Index']
avg_sa2.columns= ['Walkability Index']

#Code below is related to plotting the dataframe, and then customizing it using matplotlib.
sa3_bars = avg_sa3.sort_values('Walkability Index',ascending=False).plot(kind='bar',figsize=(8,6)) #Creating and matplotlib object
sa3_bars.set_ylabel('Walkability Index') #Setting the Y label
sa3_bars.set_xlabel('SA3 Name') #Setting the X label
sa3_fig = sa3_bars.get_figure() #Assigning the figure to a new variable for further operations
sa3_fig.tight_layout() #To fit everything in the saved image
sa3_fig.subplots_adjust(top=0.93) #Adjusting the dimensions of the plot. 
sa3_fig.suptitle('Inner Melbourne SA3 Walkability Index',fontsize=15,fontweight='bold') #Title for the figure
sa3_fig.savefig("figures/sa3_bars.png") #Saving the resulting figure as a .png image.

sa2_bars = avg_sa2.sort_values('Walkability Index',ascending=False).plot(kind='bar',figsize=(12,8))
sa2_bars.set_ylabel('Walkability Index')
sa2_bars.set_xlabel('SA2 Name')
sa2_fig = sa2_bars.get_figure()
sa2_fig.tight_layout()
sa2_fig.subplots_adjust(top=0.93)
sa2_fig.suptitle('Inner Melbourne SA2 Walkability Index',fontsize=15,fontweight='bold')
sa2_fig.savefig("figures/sa2_bars.png")

