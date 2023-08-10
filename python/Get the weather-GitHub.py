import pandas as pd #for storing the scraped weather data in a local database
from urllib.request import urlopen #to download the html from the target website
from bs4 import BeautifulSoup #to navigate the downloaded html
import datetime #to do easy date math, setting the starting and ending dates
import numpy as np #for basic math purposes
import seaborn as sns #to create basic visualizations
import matplotlib.pyplot as plt #to create basic visualizations

initDate='11/19/2013';
finDate='1/8/2017';
initDateDT=datetime.datetime.strptime(initDate,'%m/%d/%Y')
finDateDT=datetime.datetime.strptime(finDate,'%m/%d/%Y')
periodDT=(finDateDT-initDateDT).days+1
datelist = pd.date_range(initDate, periods=periodDT, freq='D').strftime('%Y%m%d')
datelist = pd.date_range(initDate, periods=periodDT, freq='D')
uniqueDates=list(datelist);
print(min(uniqueDates))
print(max(uniqueDates))

df=pd.DataFrame()

categories=['Mean Temperature', 'Max Temperature', 'Min Temperature', 'Precipitation', 
 'Wind Speed','Max Wind Speed','Max Gust Speed','Events'];

z=0;
for date in uniqueDates:
    year, month, day=date.strftime('%Y %m %d').split(' ')
    url='https://www.wunderground.com/history/airport/KEUG/%s/%s/%s/DailyHistory.html?req_city=Eugene&req_state=OR&req_statename=Oregon&reqdb.zip=97404&reqdb.magic=1&reqdb.wmo=99999' % (year, month, day)
    html = urlopen(url);
    bsObj = BeautifulSoup(html.read(), 'lxml');
    bsObj2=bsObj.find('table', class_='responsive airport-history-summary-table');
    bsObj2_list=bsObj2.find_all('td');
    x=0;
    subset=[];
    for term in categories:
        if term=='Events':
            subset3=[]
            for y in range(x,len(bsObj2_list)):
                if bsObj2_list[y].text==term:
                    subset3.append(term)
                    subset3.append(bsObj2_list[y+1].text.replace('\t','').replace('\n',''));
            subset.append(subset3)
        for y in range(x, len(bsObj2_list)):
            try:
                if bsObj2_list[y].text==term:
                    subset2=[];
                    subset2.append(term);
                    subset2.append(bsObj2_list[y+1].find(class_='wx-value').text)
                    try:
                        subset2.append(bsObj2_list[y+2].find(class_='wx-value').text)
                    except:
                        pass
                    try:
                        subset2.append(bsObj2_list[y+3].find(class_='wx-value').text)
                    except:
                        pass
                    subset.append(subset2)
            except:
                pass
    for term in subset:
        if len(term)>=3:
            subset.append([term[0]+' Average',term[2]])
        if len(term)>=4:
            subset.append([term[0]+' Record',term[3]])
    for term in subset:
        try:
            df.loc[z,term[0]]=term[1]
        except:
            pass
        df.loc[z,'Date']=date
    z=z+1

df.head()

df.loc[df['Precipitation']=='T','Precipitation']=np.nan; #I don't know what 'T' stands for, but let's remove it.
Cols=df.columns;
for col in Cols:
    try:
        df[col]=df[col].astype(float) #making anything that can be a number, a number.
    except:
        pass
df['Date_Diff']=df['Date'].apply(lambda x: (x - df.loc[:,'Date'].min()).days) #convert for daysf since start
df.loc[:,'Events'].fillna('None',inplace=True) #Fill blanks
df.loc[:,'Events']=df.loc[:,'Events'].str.replace('\xa0','None'); #fill equivalent of blanks

sns.regplot(x='Date_Diff',y='Mean Temperature',data=df, fit_reg=False)
plt.show()

sns.regplot(x='Date_Diff',y='Mean Temperature',data=df, fit_reg=False)
sns.regplot(x='Date_Diff',y='Mean Temperature Average',data=df, fit_reg=False)
plt.show()

df.to_csv('weather_data.csv',index=False)

(4000*60)/60/60



