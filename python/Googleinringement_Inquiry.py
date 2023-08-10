import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use("ggplot")
import requests
from bs4 import BeautifulSoup
import datetime as dt

df_domains = pd.read_csv("data/domains.csv") #, nrows=100000
df_TL_domains = pd.read_csv("data/160812_Top_Level_Domains.csv")
df_TL_countries = pd.read_csv("data/TL_DOMAINS_All_COUNTRIES.csv")

df_requests = pd.read_csv("data/requests.csv")# nrows=100000

#Function to create the Domain TL 

#Call the first element of the list works fine in the Test version. But as soon as you 
#you work with the large data set it creates error messages. To work aroung them I 
#transfered the func output into a tuple. Somewhere along the line there is a corrupt 
#cell. How can I find it?

import re
def toplevel_domains(xx):
    if xx == 'unknown':
        return
    try:
        return re.findall(r'\.[a-z][a-z]+$', xx)[0] #tuple
    except:
        return None

df_domains['Domain TL'] = df_domains['Domain'].apply(toplevel_domains)

#Now I want to count the number of URLs that were removed and group them by Domain TL.
df_domains_removed_Count = pd.DataFrame(df_domains.groupby('Domain TL')['URLs removed'].sum().sort_values(ascending=False))
df_domains_removed_Count = df_domains_removed_Count.reset_index()

df_domains_removed_Count.head(1)

df_TL_domains.head(1)

df_TL_countries.head(1)

#Merging the "Removed URL Count" with the "Top Level Domain Count by country" 
#and the "TL Domain types and country.
df = df_TL_domains.merge(df_domains_removed_Count, left_on='name', right_on='Domain TL')
df = df.merge(df_TL_countries, left_on='name', right_on='Domain')

del df['Unnamed: 0_x']
del df['name']
del df['Unnamed: 0_y']
del df['index']
del df['Domain']

df.info()

df['removed URLs per TLD'] = df['URLs removed'] / df['count']

df['removed URLs per TLD'] = df['removed URLs per TLD'].astype(float)

df.sort_values(by='removed URLs per TLD', ascending=False).head(5)

df_countries = df[df['Type'] == 'country-code']

df_countries.sort_values(by='removed URLs per TLD', ascending=False)

df_countries.to_csv('Removed_URLs_by_Country.csv')

# .com
# .net
# Country Codes
# Rest

df.head()

#Total first:
Total = df['URLs removed'].sum()

#How may links where .com links?
com_sum = df[df['Domain TL'] == '.com']['URLs removed'].sum()

#How may links where .com links?
net_sum = df[df['Domain TL'] == '.net']['URLs removed'].sum()

TLcountry_sum = df_countries['URLs removed'].sum()

Rest = Total - com_sum - net_sum - TLcountry_sum

#Creating a dictionary with these values

Pie_dict = [{'type': '.com', 'Count': com_sum}, {'type': '.net', 'Count': net_sum}, {'type': 'Other Generic',             'Count': Rest}, {'type': 'Country Domains', 'Count': TLcountry_sum}] 

TL_cat = pd.DataFrame(Pie_dict)

#Creasting Percentage column
TL_cat['Percent'] = round(TL_cat['Count'] / TL_cat['Count'].sum() * 100)

fig, ax = plt.subplots(figsize =(5,5), facecolor='WhiteSmoke')
explode = (0, 0, 0, 0.15)
labels = TL_cat['type']

TL_cat.plot(kind='pie', y='Count', ax=ax, explode=explode, autopct='%1.f%%', labels=labels, legend=False)
ax.set_ylabel("")
ax.set_title("Generic versus Country Top Level Domains", fontname='DIN Condensed', fontsize=24)
plt.savefig('TL_pie.pdf', transparent=True, bbox_inches='tight')

df_domains['Domain'].value_counts().head(10)

companies = df_domains['Domain'].value_counts().head(10)

fig, ax = plt.subplots(figsize =(7,4), facecolor='WhiteSmoke')
companies.sort_values().plot(kind='barh')

ax.set_ylabel(ylabel='')
ax.set_xlabel(xlabel='')
ax.set_axis_bgcolor("WhiteSmoke")

plt.tick_params(
    #axis='x',
    top='off',
    which='major',
    left='on',
    right='off',
    bottom='off',
    labeltop='off',
    labelbottom='on',
    labelright='off',
    labelleft='on')

plt.savefig('companies_bar.pdf', transparent=True, bbox_inches='tight')

df_requests.head(1)

dt.datetime.strptime('07/06/2015 10:58:27 AM', '%m/%d/%Y %I:%M:%S %p')
#datetime.datetime(2015, 7, 6, 0, 0)
parser = lambda date: pd.datetime.strptime(date, '%m/%d/%Y %H:%M:%S')

df_requests = pd.read_csv("data/requests.csv", low_memory=False, parse_dates=[1], dtype=str)

df_requests.index = df['Date']

fig, ax = plt.subplots(figsize=(10,4), facecolor='WhiteSmoke')
df.resample('M')['Request ID'].count().plot(y='URLs removed')
ax.set_ylabel(ylabel='')
ax.set_xlabel(xlabel='')
ax.set_axis_bgcolor("WhiteSmoke")

plt.tick_params(
    #axis='x',
    top='off',
    which='major',
    left='on',
    right='off',
    bottom='off',
    labeltop='off',
    labelbottom='on',
    labelright='on',
    labelleft='off')
plt.savefig('time_plot.pdf', transparent=True, bbox_inches='tight')

df_domains[df_domains['Domain TL'] == '.ch']['URLS removed'].values_count()

df_countries.head()

df_countries[df_countries['Country Name'] == 'Switzerland']





ch_df_with_dates.resample('D').sum().plot(y='URLs removed_x', label='Removed URLs', figsize=(10,4))

ch_df_with_dates

uploadablech_df = ch_df_with_dates[ch_df_with_dates['Domain'] == 'uploadable.ch']

uploadablech_df

uploadablech_df.resample('W').sum().plot(y='URLs removed_x', label='Removed URLs', figsize=(10,4))



com_df = df[df['Top Level Domains'] == '.com']

com = df[df['Top_Level'] == '.com']



