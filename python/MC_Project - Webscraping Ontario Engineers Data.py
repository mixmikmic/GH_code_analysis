# import the dependencies
#import json

# This is needed to create a lxml element object that has a css selector
#from lxml.etree import fromstring

# The requests librart
from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
import time

#==============================================================================
# Extract Discipline Option numbers
#==============================================================================
disc = requests.get('http://forum.peo.on.ca/cgi-bin/EPIM_Search/EPIM_Form_Search.do')
options = bs(disc.text, "lxml").find_all('option')[1].parent
options_list = options.find_all('option')
disc_vals = [x['value'] for x in options_list]
disc_names = [x.text for x in options_list]
disc_val_and_names = dict(zip(disc_vals,disc_names))
disc_vals = disc_vals[1:]

#==============================================================================
# Extract RegIDS and names
#==============================================================================
regs_all = []
regs_sector_all = []
for val in disc_vals:
    sector = disc_val_and_names[val]
    get_reg_html = requests.get('http://forum.peo.on.ca/cgi-bin/EPIM_Aptify/EPIM_Process_Search.do?RegID=&FirstName=&LastName=&DiscCode={}&City=&Employer=&CofA=&Postal=&Chapter='.format(val))
    regs = [x['value'] for x in bs(get_reg_html.text,"lxml").find_all('option')]
    regs_all = regs_all + regs
    regs_sector = [disc_val_and_names[val] for x in range(0,len(regs))]
    regs_sector_all = regs_sector_all + regs_sector
    
regsDF = pd.DataFrame({'RegIDs':regs_all,'Sector':regs_sector_all})
regsDF = regsDF.drop_duplicates('RegIDs')
regsDF = regsDF.set_index('RegIDs')

#==============================================================================
# Extract data from each regID
#==============================================================================
MemberInfo={}
EmployeeInfo={}

attempts = 74
completed = 0
remaining = len(regsDF['Sector'])
total = len(regsDF['Sector'])
#you probably need to write a while loop in here... so that it keeps trying until list is complete. See link below:
#http://stackoverflow.com/questions/6380290/python-if-error-raised-i-want-to-stay-in-script     
for attempt in range(1,attempts):
    if remaining > 1000:
        for reg in regsDF.index[completed:completed+1000]:
            data = requests.get('http://forum.peo.on.ca/cgi-bin/EPIM_Aptify/EPIM_Process.do?RegID={}'.format(reg))
            data_s = bs(data.text, "lxml")
           
            meminfoL = [x.text for x in data_s.find_all('replacecontent')[2].find_all('td',{'class':'left'})]
            meminfoR = [x.text for x in data_s.find_all('replacecontent')[2].find_all('td',{'class':'cell'})]
            MemberInfo[reg]=dict(zip(meminfoL,meminfoR))
            
            empinfoL = [x.text for x in data_s.find_all('replacecontent')[6].find_all('td',{'class':'left'})]
            empinfoR = [x.text for x in data_s.find_all('replacecontent')[6].find_all('td',{'class':'cell'})]
            EmployeeInfo[reg]=dict(zip(empinfoL,empinfoR))
    else:
        for reg in regsDF.index[completed:total-1]:
            data = requests.get('http://forum.peo.on.ca/cgi-bin/EPIM_Aptify/EPIM_Process.do?RegID={}'.format(reg))
            data_s = bs(data.text, "lxml")
            meminfoL = [x.text for x in data_s.find_all('replacecontent')[2].find_all('td',{'class':'left'})]
            meminfoR = [x.text for x in data_s.find_all('replacecontent')[2].find_all('td',{'class':'cell'})]
            MemberInfo[reg]=dict(zip(meminfoL,meminfoR))
            empinfoL = [x.text for x in data_s.find_all('replacecontent')[6].find_all('td',{'class':'left'})]
            empinfoR = [x.text for x in data_s.find_all('replacecontent')[6].find_all('td',{'class':'cell'})]
            EmployeeInfo[reg]=dict(zip(empinfoL,empinfoR))
    
    completed = len(EmployeeInfo)
    remaining = total - completed
    print("Attempt {} complete! So far so good...".format(attempt))
    time.sleep(180)

#==============================================================================
# Put data into single DataFrame
#============================================================================== 
MemberInfoDF = pd.DataFrame(MemberInfo).transpose()
EmployInfoDF = pd.DataFrame(EmployeeInfo).transpose()

AllEngineers = pd.concat([MemberInfoDF,EmployInfoDF], axis=1)

#==============================================================================
# Put data into excel
##============================================================================== 
from pandas import ExcelWriter
writer = ExcelWriter('AllEngineers.xlsx')
AllEngineers.to_excel(writer,'AllEngineers')
writer.save()


#A BUNCH OF PPL ARE IN THIS LIST TWICE!

#regsDF[0].nunique()
#Out[226]: 72916
#
#len(regsDF[0])
#Out[227]: 78394
#ISOLATE DUPLICATES... turns out, if they have multiple disciplines they are in both categories!
#regsDF['dups']=regsDF.duplicated('test')

#r = requests.get('http://forum.peo.on.ca/cgi-bin/EPIM_Aptify/EPIM_Process.do?RegID=214205|2000')
#
#r.status_code # 200
#
#r.text # your xml data
#
#rsoup = bs(r.text, "lxml")
#agr_s = bs(agr.text, "lxml")
#bio_s = bs(bio.text, "lxml")


