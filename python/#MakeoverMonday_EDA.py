#import libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

import warnings
warnings.filterwarnings("ignore")

get_ipython().magic('matplotlib inline')

#change directory and import dataset
os.chdir('D:\\MakeoverMondayDataFiles')
data=pd.read_csv('data.csv',encoding='latin1')
data.head()

#get data info
data.info()

#check the nulls visually
msno.matrix(data)

#see first entry when we started pulling data
data.iloc[-1,:]

#total number of entries from data.world
Data_dot_world_entries=data[data.Name=='datadotworld'].shape[0]
print('total entries of data.world: {}'.format(Data_dot_world_entries))

#filtering out data.world entries, update csv, and check totals
data=data[data.Name!='datadotworld']
data.to_csv('data.csv',index=False)
data.info()

#check the last entry again
data.iloc[-1,:]

len(data.Name.unique())

#Participant frequency of once vs more than once.
post_freq=data.Name.value_counts(ascending=False)

#Create ecdf function
def ecdf(data):
    xs=sorted(data)
    ys=np.arange(1,len(data)+1)/len(data)
    return xs,ys

#check ecdf for percentage difference
xs,ys=ecdf(post_freq)
plt.figure(figsize=(18,8))
plt.plot(xs,ys, marker='.',linestyle='none')
plt.margins(0.02)
plt.xticks(xs)
plt.title('ECDF of Participants')
plt.xlabel('Number of Posts')
plt.ylabel('Percentage of Participants')
plt.show()

outliers=(post_freq>25)
outliers[:5]

#created list of totals from above
post_totals=sorted(post_freq.unique().tolist())
post_totals

#Reloading data on 3/25/18 in first two cells and post_freq
#Now to get a list of the indexes (aka names of 'participants')
participants_grouped=[]
for total in post_totals:
    participants_grouped.append(post_freq[post_freq==total].index.tolist())

participants_grouped

#creating dictionary of post_totals and participants_grouped (added 3/26/18, includes tableau and tableau public
#prior to filtering)
participant_dict=dict(zip(post_totals,participants_grouped))

#testing with keys
participant_dict[6]

#getting totals for each person that posted x amount of times.
for i in range(len(participants_grouped)):
    print('Number of people that posted {}: {}'.format(post_totals[i],len(participants_grouped[i])))

#import modules and libraries to parse tweet text
import re
import nltk
from nltk.tokenize import TweetTokenizer

#create instance of tweettokenizer and then tokenize the tweets from the data.csv
twtoken=TweetTokenizer()
data['TextTokenized']=data['Text'].apply(twtoken.tokenize)
data.head()

#filtering for Rodrigo Calloni since makeovermonday participants were asked to put the week in their tweet to label their 
#participation for that week.
rodrigo=data[data.Name=='Rodrigo Calloni'][data.Text.str.contains(r'week\s+\d')==True]
rodrigo

#get number of entries
rodrigo.shape[0]

#see text
rodrigo['Text']

#check for tableau to see if they had any promotional entries
check=data[data.TwitterHandle=='tableau']
check.head()

#see number of entries
print('Number of Entries from Tableau: {}'.format(check.shape[0]))

#Import falsetweets csv
falsetweets=pd.read_csv('falsetweets.csv',encoding='latin1')
falsetweets.head()

#append check to falsetweets and confirm additions
falsetweets=falsetweets.append(check.iloc[:,:9],ignore_index=True)
falsetweets.tail(9)

#Filter out tableau tweets
print(data.shape)
data=data[data.TwitterHandle!='tableau']
print(data.shape)

#checking for tweets from tableaupublic
data[data.TwitterHandle=='tableaupublic']

#droping tableaupublic tweets from data and updating csvs
falsetweets=falsetweets.append(data[data.TwitterHandle=='tableaupublic'].iloc[:,:9],ignore_index=True)
data=data[data.TwitterHandle!='tableaupublic']

#check if tweets were
falsetweets.tail()

#checking for number of unique names
len(data.Name.unique())

data[data.Name=='Doc Kevin Lee Elder']

#Filter out DocKevinElder since his tweets are essentially retweets of participants
falsetweets=falsetweets.append(data[data.TwitterHandle=='DocKevinElder'].iloc[:,:9],ignore_index=True)
data=data[data.TwitterHandle!='DocKevinElder']

falsetweets.tail(8)

#Update csvs
falsetweets.to_csv('falsetweets.csv',index=False)
data.to_csv('data.csv',index=False)

#continuing on 3/27/18
#filter out slt tableau user group 
falsetweets=falsetweets.append(data[data.Name=='STL Tableau User Grp'].iloc[:,:9],ignore_index=True)
data=data[data.Name!='STL Tableau User Grp']

falsetweets.tail(10)

#filter out y94
falsetweets=falsetweets.append(data[data.Name=='Y94'].iloc[:,:9],ignore_index=True)
data=data[data.Name!='Y94']

falsetweets.tail(10)

participant_dict[2]

#filter out equal 2030
falsetweets=falsetweets.append(data[data.Name=='EqualMeasures2030'].iloc[:,:9],ignore_index=True)
data=data[data.Name!='EqualMeasures2030']
falsetweets.tail(11)

#filter out fenceanddeck connection
falsetweets=falsetweets.append(data[data.Name=='Fence&DeckConnection'].iloc[:,:9],ignore_index=True)
data=data[data.Name!='Fence&DeckConnection']
falsetweets.tail(11)

#function for updating falsetweets and filtering out data
def update_and_filter(name,ft,df):
    false=ft.append(df[df.Name==name].iloc[:,:9],ignore_index=True)
    df=df[df.Name!=name]
    return false, df

#testing function with the information lab
falsetweets,data=update_and_filter('The Information Lab',falsetweets,data)

falsetweets.tail(11)
#successful

#filter'AntonioASalonForHair'
falsetweets,data=update_and_filter('AntonioASalonForHair',falsetweets,data)

participant_dict[1]

data[data.Name=='Data Science Renee']

#check shape
data.shape

#drop duplicates
data=data.drop_duplicates(subset='Text',keep='last')

#see now size of data after droping duplicates
data.shape

#create list of false participants to run through function
fake_participants=['Information Lab Irl','Just A Guy Who Supports His Girls Dreams??',
                   'Infographics News','PussyandPooch','FASHION x ON H St','favkart','Tonsorial HairDesign',
                  'TopCats Cheerleaders','Spotlight Group','thefringefestival','ClickInsight','makeup.by.janneth',
                  'UF Project Makeover','Dr. Desiree Yazdan','JT Jack-fm','Level 10 Studio','Advent',
                  'North Haven Library','Soar Mill Cove Hotel','SupaShirts','Heritage Works DBQ',
                  'Clinica Joelle Egypt','Colors On Parade','Harpak Home Shopping','Eric Fisher Salon','The Ultimate Shave',
                  'Bath Splash Showroom','QVC UK','Roll Former USA','Zeeyna Cubana SharkBait Haddad','Rutland GutterSupply',
                  'Zentraedi Online','INFUSE Humber','FusionCharts',"J Michael'sSpa&Salon","Fred's Carpets Plus",
                   'Rogue Penguin','Sheryl P. Denker PhD','RE/MAX Commonwealth','UC Fitness Apparel','BeachBlvd FleaMarket',
                  'TransMedia Group','Saturday Night Live - SNL','Supercuts UK','Spa La Posada','Camden Group',
                   'Audacious Beauty Cosmetics Calgary','KDB BEAUTY SHOP','Mark Olson DDS FICOI','Dr. Catrise Austin',
                  'Randy Olson','Dayside Windows','Affordable Granite','skylitserenade','InstituteComputation',
                  'McC5','ConsultantsClub','Tweak It Shop',"Rock'N'Veg",'YourGoldenTicketBlog','Dr. Trent Tucker',
                  'The Oyster Shed','Danielle Kelly Art','Renaissance Dental','Kevin B. Sands, DDS','Fashion Brunch Club',
                  'Data Science Renee']

#run list of names through function
for name in fake_participants:
    falsetweets,data=update_and_filter(name,falsetweets,data)

#print new sizes of data and falsetweets
print(data.shape)
print(falsetweets.shape)

#Update csvs
falsetweets.to_csv('falsetweets.csv',index=False)
data.to_csv('data.csv',index=False)



