from selenium import webdriver #selenium is used to interact with the webpage, so the program can 'click' buttons.
import pandas as pd #the data will be saved locally as a csv file. Pandas is a nice way to write/read/work with those files.
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary #This will let the program open the webpage on a new Firefox window.
from bs4 import BeautifulSoup #BeautifulSoup is used to parse the HTML of the downloaded website to find the particular information desired.
import time #I will need to delay the program to give the webpage time to open. time will be used for that.
import sys #This is only used to assign a location to my path. The location where I have a needed file for selenium.

sys.path
sys.path.append('/path/to/the/example_file.py')
sys.path.append('C:\\Users\\Kyle\\Documents\\Blog Posts\\EugeneEMSCalls')

Months={'Jan':'01','Feb':'02','Mar':'03','Apr':'04', 'May':'05', 'Jun':'06', 'Jul':'07', 'Aug':'08', 'Sep':'09', 'Oct':'10', 'Nov':'11', 'Dec':'12'}

#url of the website with the call log data
url='http://coeapps.eugene-or.gov/ruralfirecad'

binary = FirefoxBinary('C:\\Program Files (x86)\\Mozilla Firefox\\firefox.exe')
driver = webdriver.Firefox(firefox_binary=binary)
driver.get(url)

#quick check to see if selenium correctly got to the page. 
#this searches the HTML of the page for the HTML element id'd as 'callSummary',
#and prints the text.
summary=driver.find_element_by_id('callSummary').text
print(summary)

endIt=0;
while endIt==0:
    calendarOptions=driver.find_element_by_id('calendar').text.split()
    monthsDays=[];
    for elm in calendarOptions:
        try:
            potlDay=int(elm)
            if potlDay<=31:
                monthsDays.append(elm)
        except:
            pass
    for day in monthsDays:
        driver.find_element_by_link_text(day).click()
        time.sleep(1) #giving firefox time to open.
        summary=driver.find_element_by_id('callSummary').text;
        dateData=summary[summary.index('on')+3:summary.index(':')].replace(',','').split(' ');
        if len(dateData[1])==1:
            dateData[1]='0'+dateData[1]
        date=int(dateData[2]+str(Months[dateData[0]])+dateData[1]);
        if int(date)==20161201: #End date is located here.
            endIt=1;
            break
        html = driver.page_source;
        soup = BeautifulSoup(html,'lxml'); #using BeautifulSoup to find the call logs.
        EMSdata=soup.find('table', class_='tablesorter');
        colNames1=EMSdata.thead.findAll('th') #recording the column names.
        colNames2=[];
        data1=[]
        for x in range(0,len(colNames1)):
            colNames2.append(colNames1[x].string.strip()) #saving each column value.
            data1.append([])
        for row in EMSdata.findAll("tr"): #saving the individual call log data.
            cells = row.findAll('td')
            if len(cells)==len(colNames1):
                for y in range(0,len(cells)):
                    data1[y].append(cells[y].string.strip())
        EMSdata1=pd.DataFrame(); #initializing a data frame to save 1 days worth of calls.
        for x in range(0,len(colNames2)):
            EMSdata1[colNames2[x]]=data1[x]
        EMSdata1['Date']=date;
        try:
            EMSdata1.to_csv('%s.csv'%(EMSdata1.loc[0,'Date']),index=False) #saving csv file of daily call logs.
        except:
            pass
        time.sleep(1) #giving time to save csv before moving on.
    if endIt==0:
        driver.find_element_by_link_text('Prev').click()



