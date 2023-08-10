import pandas as pd #to store our scraped data
from urllib.request import urlopen #to download page contents from a given valid url
from bs4 import BeautifulSoup #to navigate around the page's html
import datetime #to make stepping through dates easy during scraping

"""https://www.wunderground.com/history/airport/KEUG/2017/01/03/
DailyHistory.html?req_city=Eugene&req_state=OR&req_statename=Oregon&reqdb.zip=
97404&reqdb.magic=1&reqdb.wmo=99999""";

#Added ease by using the package datetime to iterate through the days
day=datetime.date.today();
one_day = datetime.timedelta(days=1);
#the rows I'll scrape from the table in the web page:
temps=['Mean Temperature','Max Temperature','Min Temperature']
#Initialize the pandas dataframe with a column for each value I will scrape from each page.
df=pd.DataFrame(columns=['Date','Mean Temperature','Mean Temperature Average',
                         'Max Temperature','Max Temperature Average','Max Temperature Record',
                         'Min Temperature','Min Temperature Average','Min Temperature Record'])
#Number of days worth of data I will scrape, starting today and working backwards:
for t in range(0,30):
    #creating new url for each pass of the for-loop
    dayStr=day.strftime('%Y %m %d')
    yr, mon, da=dayStr.split(' ')
    day=day-one_day
    url="""https://www.wunderground.com/history/airport/KEUG/%s/%s/%s/DailyHistory.html?req_city=Eugene&req_state=OR&req_statename=Oregon&reqdb.zip=97404&reqdb.magic=1&reqdb.wmo=99999""" %(yr,mon,da);
    #opening and parsing the html of the page
    html = urlopen(url);
    soup=BeautifulSoup(html.read(),'html.parser');
    #finding a subsection of the html. A table which contains all the data I desire
    soup2=soup.find('table', class_='responsive airport-history-summary-table');
    #locating most rows in the table
    classSet=soup2.find_all(class_='indent');
    vals=[];
    #iterating through the rows of the table to find the ones with data I desire
    for y in range(0,len(classSet)):
        a=classSet[y]
        cat=str(a.contents).replace('[<span>','').replace('</span>]','');
        if cat in temps:
            for x in range(0,10):
                if 'wx-value' in str(a):
                    b=a.find(class_='wx-value')
                    c=str(b.contents).replace("['","").replace("']","");
                    vals.append(int(c))
                try:
                    a=a.next_sibling
                except:
                    break
    df.loc[t,'Date']=yr+mon+da
    df.iloc[t,1:]=vals
df



