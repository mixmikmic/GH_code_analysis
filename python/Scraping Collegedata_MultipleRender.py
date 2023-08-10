get_ipython().magic('matplotlib inline')
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")

import requests
import time
from bs4 import BeautifulSoup
import re
import TIdatabase as ti

college_ids=['Princeton', 'Harvard', 'Yale', 'Columbia', 'Stanford', 'UChicago', 'MIT', 'Duke', 'UPenn', 'CalTech', 'JohnsHopkins', 'Dartmouth', 'Northwestern', 'Brown', 'Cornell', 'Vanderbilt', 'WashU', 'Rice', 'NotreDame', 'UCB', 'Emory', 'Georgetown', 'CarnegieMellon', 'UCLA', 'USC']
college_urls=[111, 444, 244, 399, 781, 327, 186, 1026, 67, 706, 1509, 403, 1803, 163, 787, 1562, 1720, 731, 1774, 1090, 1039, 1182, 204, 1093, 1138]
college_id_dict=dict(zip(college_ids,college_urls))
baseurl='http://www.collegedata.com/cs/admissions/'
tracker_url='admissions_tracker_result.jhtml?schoolId='
student_url='admissions_profile_view.jhtml?profileName='

## Source: https://webscraping.com/blog/Scraping-multiple-JavaScript-webpages-with-webkit/
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtWebKit import *

class Render(QWebPage):  
    def __init__(self, urls):
        self.app = QApplication(sys.argv)  
        QWebPage.__init__(self)  
        self.loadFinished.connect(self._loadFinished)  
        self.urls = urls 
        self.profileList=set()
        self.crawl()  
        self.app.exec_()  
      
    def crawl(self):  
        if self.urls:  
            url = self.urls.pop(0)    
            self.mainFrame().load(QUrl(url))  
        else:  
            self.app.quit()  
        
    def _loadFinished(self, result): 
        frame = self.mainFrame()  
        url = str(frame.url().toString())  
        html = frame.toHtml()  
        self.scrape(url, html)
        self.crawl()
    
    # Once we have the html code with processed Javascript, we can parse it to find the profile names
    def scrape(self,url, html):
        soup = BeautifulSoup(str(html.toAscii()),'html.parser')   # we create a beautiful soup object   
        profiles=soup.find_all("a",href=re.compile(r"enterProfileByName")) # find all <a> tags that contain the string 'enterProfileByName'
        for p in profiles:
            self.profileList.add(p.get("href").split("'")[1]) #the profile name is between apostrophe's

#urls=[]
#schoolid=111
#schoolurl=baseurl+tracker_url+str(schoolid)+'&classYear='
#for year in range(2009,2011):
#    urls.append(schoolurl+str(year))
#r=Render(urls)
#print len(r.profileList)

get_ipython().run_cell_magic('time', '', "urls=[]\nfor school in college_id_dict:\n    schoolurl=baseurl+tracker_url+str(college_id_dict[school])+'&classYear='\n    for year in range(2009,2020):\n        urls.append(schoolurl+str(year))\nr=Render(urls)\nprint 'We found ', len(r.profileList), 'students'")

# these are bad profiles without any information
r.profileList.remove('orangecat')
r.profileList.remove('j7Wa4')
r.profileList.remove('EDz3k')
print 'We found ', len(r.profileList), 'students'

columns_student = ['classrank', 'admissionstest','AP','averageAP','SATsubject', 'GPA', 'GPA_w', 'program','schooltype',
            'intendedgradyear', 'addInfo', 'canAfford', 'female', 'MinorityGender','MinorityRace','international',
           'firstinfamily','sports','artist', 'workexp']
columns_uni = ['collegeID','earlyAppl','visited','alumni', 'outofstate', 'acceptStatus','acceptProb']

# Indicators for gender, corresponding to dataframe column 'female':
genderdict = {'Male': -1, 'Female': 1} 
#Indicator for the type of high school for the dataframe column 'schooltype':
highschooldict = {'Public': -1, 'Private': 1, 'Parochial': 1, 'Home-Schooled': 1} 
# A list of words we associate with underrepresented minority races for the dataframe column 'MinorityRace'
minoritylist = ['african', 'hispanic', 'latin','indian', 'native', 'black', 'mexican','puerto','alaska','hawai','pacific island']
# A list of SAT scores from https://www.act.org/solutions/college-career-readiness/compare-act-sat/ corresponding to ACT composite scores between 36 and 11
sats=[1600, 1560, 1510, 1460, 1420, 1380, 1340, 1300, 1260, 1220, 1190, 1150, 1110, 1070, 1030, 990, 950, 910, 870, 
      830, 790, 740, 690, 640, 590, 530]
# A dictionary to translate an ACT composite score to an SAT CR+M score.
act2satdict=dict(zip(range(36,10,-1),sats))
# General indicator for boolean columns in the webpage. For example the column 'Athlete' of 'Alumni'
booleandict={'': -1, 'X': 1}
# Indicators for the admission status corresponding to dataframe column 'acceptStatus'
statusdict={'Will Attend': 1, 'Accepted': 1, 'Applied': 0, 'Deferred': -1, 'Denied': -1, 'Not Applied': 0, 'Wait-Listed': -1, 'Withdrawn': 0, 'Pending': -1}
# List of university names as used on collegedata.com
uni_list=['Princeton University', 'Harvard College', 'Yale University', 'Columbia University', 'Stanford University', 'University of Chicago', 'Massachusetts Institute of Technology', 'Duke University', 'University of Pennsylvania', 'California Institute of Technology', 'Johns Hopkins University', 'Dartmouth College', 'Northwestern University', 'Brown University', 'Cornell University', 'Vanderbilt University', 'Washington University in St. Louis', 'Rice University', 'University of Notre Dame', 'University of California, Berkeley', 'Emory University', 'Georgetown University', 'Carnegie Mellon University', 'University of California, Los Angeles', 'University of Southern California']
# Dictionary to translate the university name used on collegedata.com to the university name used in our dataframe
uni_name_dict = dict(zip(uni_list, college_ids))
# List of states for each university
uni_state=['NJ', 'MA', 'CT', 'NY', 'CA', 'IL', 'MA', 'NC', 'PA', 'CA', 'MD', 'NH', 'IL', 'RI', 'NY', 'TN', 'MO', 'TX', 'IN', 'CA', 'GA', 'DC', 'PA', 'CA', 'CA']
# Dictionary get the state of a university
uni_state_dict = dict(zip(uni_list,uni_state))
# Dictionary to translate a state to its abbreviation 
states_dict={'Alabama': 'AL',
 'Alaska': 'AK',
 'American Samoa': 'AS',
 'Arizona': 'AZ',
 'Arkansas': 'AR',
 'California': 'CA',
 'Colorado': 'CO',
 'Connecticut': 'CT',
 'Delaware': 'DE',
 'District of Columbia': 'DC',
 'Florida': 'FL',
 'Georgia': 'GA',
 'Guam': 'GU',
 'Hawaii': 'HI',
 'Idaho': 'ID',
 'Illinois': 'IL',
 'Indiana': 'IN',
 'Iowa': 'IA',
 'Kansas': 'KS',
 'Kentucky': 'KY',
 'Louisiana': 'LA',
 'Maine': 'ME',
 'Maryland': 'MD',
 'Massachusetts': 'MA',
 'Michigan': 'MI',
 'Minnesota': 'MN',
 'Mississippi': 'MS',
 'Missouri': 'MO',
 'Montana': 'MT',
 'National': 'NA',
 'Nebraska': 'NE',
 'Nevada': 'NV',
 'New Hampshire': 'NH',
 'New Jersey': 'NJ',
 'New Mexico': 'NM',
 'New York': 'NY',
 'North Carolina': 'NC',
 'North Dakota': 'ND',
 'Northern Mariana Islands': 'MP',
 'Ohio': 'OH',
 'Oklahoma': 'OK',
 'Oregon': 'OR',
 'Pennsylvania': 'PA',
 'Puerto Rico': 'PR',
 'Rhode Island': 'RI',
 'South Carolina': 'SC',
 'South Dakota': 'SD',
 'Tennessee': 'TN',
 'Texas': 'TX',
 'Utah': 'UT',
 'Vermont': 'VT',
 'Virgin Islands': 'VI',
 'Virginia': 'VA',
 'Washington': 'WA',
 'West Virginia': 'WV',
 'Wisconsin': 'WI',
 'Wyoming': 'WY',
  'Other': 'Other'}

# getFromDict: general function that looks up a key in a dictionary and returns the value if available and 0 if not available
#              input:      dictionary = any dictionary for indicators
#                          text = key to look up in dictonary
#              output:     if the dictionary has text as input, output = dictionary[text], else output=0
def getFromDict(dictionary,text):
    if dictionary.has_key(text):
        return dictionary[text]
    else:
        return 0
# isMinority: check if a string contains one of the words associated with minorities
#              input:      text = user-supplied string describing a race
#              output:     output = 1 if the race is considered a minority, else output = -1 
def isMinority(text):
    for m in minoritylist:
        if m in text:
            return 1
    return -1
# getScores: obtain a metric from scores, given a list of html strings containing scores 
#              input:      scores = list of html strings containing scores
#                          fun = function to apply to the scores
#              output:     metric obtained by applying fun to the list of numerical scores
#                          and the number of valid scores found in the list
def getScores(scores,fun):
    scores = [s.get_text().strip() for s in scores] # clean the strings
    while '' in scores: 
        scores.remove('') # remove all empty strings
    if len(scores)>0:
        scores=[int(s) for s in scores] # create list of actual numerical scores
        return fun(scores), len(scores)
    else:
        return None, 0
# getAdmissionTestScore: obtain one single admission test score, given SAT and ACT scores
#              input:      doc = list of html strings containing the SAT scores (CR, M and W) and the ACT score
#              output:     a single score derived from the combination of available admission test scores
def getAdmissionTestScore(doc):
    satCRM,dummy = getScores(doc[0:2],sum) # Get the sum of the SAT CR and M scores
    satW,dummy = getScores([doc[2]],np.max) # Get the SAT W score
    act,dummy = getScores([doc[4]],np.max) # Get the ACT score
    if act == None: # If no ACT score available, we just use the total SAT score
        return (satCRM+satW)
    elif satCRM==None: # if no SAT CR + M score, we need to replace it with the ACT score
        if satW==None: # if also no SAT W score available, there is a problem ( but this is never the case )
            print "Warning: no SAT writing score"
        return act2satdict[act]+satW # Convert ACT to an SAT score and combine with the SAT W score
    else: # if all scores available, we use the maximum of the translated ACT score and SAT CR+M score
        return (max(act2satdict[act],satCRM)+satW)
# getCanAfford: detect the indicator 'canAfford'
#              input:      text = string from the html that indicates whether a student applied for financial support
#              output:     output =1 if the student can probably afford tuition, output = -1 if the student applied for financial support, output =0 if information unavailable
def getCanAfford(text):
    if 'Yes' in text:
        return -1
    elif 'No' in text:
        return 1
    else:
        return 0
# removePunct: general function to clean up string-based columns
#              input:      text = string to clean
#              output:     cleaned string
def removePunct(text):
    text = re.sub(r'([^\s\w]|_)+'," ",text)
    return re.sub('\s+',' ',text).encode('latin-1')

def getColumnValues(soup):
    # initialize the dictionary and list
    values=dict(zip(columns_student,[None for i in range(len(columns_student))]))
    applications=[]
    # We start with the general information box at the top which includes class year, gender and ethnicity
    doc= soup.find("div",{"class": "general"})
    values['intendedgradyear'] = int(re.findall(r'\d{4}',doc.find("h1").get_text().split('Class of')[-1])[0]) # CLASS YEAR
    doc = doc.find_all("span")
    values['female']= getFromDict(genderdict,doc[0].get_text().strip()) # GENDER
    values['MinorityGender']= 1 if values['female']==0 else -1 # Minority gender if no gender found
    values['MinorityRace'] = isMinority(doc[1].get_text().strip().lower()) # MINORITY RACE
    values['program'] = removePunct(doc[2].get_text().strip()) # PROGRAM
    # Now we look at the academics box which includes GPA and high school info
    doc = soup.find("div", {"class": "academicswrap"}).find_all("span")
    values['schooltype']=getFromDict(highschooldict,doc[0].get_text().strip()) # SCHOOL TYPE
    state=getFromDict(states_dict,doc[1].get_text().strip()) # save the state of the student
    values['international'] = 1 if state=='Other' else -1 # INTERNATIONAL indicator
    values['GPA'] = float(doc[3].get_text()) # unweighted GPA
    values['GPA_w'] = float(doc[4].get_text()) if doc[4].get_text().strip()!='' else None # Weighted GPA
    # Next, we go to the test score box which includes SAT, ACT and AP info
    values['admissionstest'] = getAdmissionTestScore(soup.find("div", {"class": "testscorewrap"}).find_all("td")) #Admissions test
    values['SATsubject'] = len(soup.find("caption",text="SAT Subject Test Scores").next_sibling.next_sibling.find_all("tr")) # Number of SAT SUBJECT
    ap_num = len(soup.find("caption",text="AP Examinations").next_sibling.next_sibling.find_all("tr")) # Number of AP's 
    values['AP']=ap_num #AP
    if ap_num>0:
        doc = soup.find("caption",text="AP Examinations").next_sibling.next_sibling.find_all("td")
        values['averageAP'],values['AP']= getScores(doc,np.mean) # AVERAGE AP score
    # Every webpage also has three text fields for any additional information, which we just save in the 'addInfo' column
    doc = soup.find_all("div", {"class": "word"})
    doc = [d.get_text().strip() for d in doc]  
    values['addInfo']= removePunct(doc[0]+doc[1]+doc[2]) # Additional info
    # Next: the colleges applied to and the admission results for the admissions table
    doc = soup.find("table", {"class": "collchoice"})
    collegelist = doc.find("tbody").find_all("tr") # every university is a row in a table
    for c in collegelist:
        uni = c.find("th").find("span").get_text().strip() #get university name
        if uni in uni_list: # if the university is one of our 25 universities
            unirow = dict(zip(columns_uni,[None for i in range(6)])) #initialize dictionary
            unirow['collegeID']=uni_name_dict[uni] # get University ID
            doc=c.find_all("td", {"class": "center"})
            unirow['earlyAppl']=booleandict[doc[0].get_text().strip()] # Early Admission indicator
            unirow['alumni']=booleandict[doc[1].get_text().strip()] # Alumni/Legacy indicator
            if values['sports']==None or values['sports']==0: 
                values['sports']=booleandict[doc[2].get_text().strip()] # Athlete indicator
            doc = doc[2].next_sibling.find_next("span")
            unirow['acceptStatus']=getFromDict(statusdict,doc.get_text().strip()) # Admission status indicator
            if values['canAfford']==None or values['canAfford']==0:
                values['canAfford']=getCanAfford(doc.find_next("span").get_text().strip()) # can Afford indicator
            unirow['outofstate']= -1 if state==uni_state_dict[uni] else 1 #compare student state to university state for OUT OF STATE indicator
            applications.append(unirow) # add dictionary to list of applications
    return values, applications

get_ipython().run_cell_magic('time', '', '# Remove old dataframes if they still exists\nif (\'students\' in locals()): \n    students.cleanup()\n    del students\nif (\'applForm\' in locals()): del applForm\n# initialize new dataframes\nstudents=ti.Student()\ncolleges = ti.College()\napplForm = ti.ApplForm()\nfor p in r.profileList: # For each profile name\n    profile_url=baseurl+student_url+p # create url\n    soup=BeautifulSoup(requests.get(profile_url).text,\'html.parser\') #get html\n    # Check for empty webpage\n    if soup.find("div", {"class": "academicswrap"})==None: \n        print p, \' not Found\'\n        continue\n    # Get information\n    newrow, applications = getColumnValues(soup)\n    # insert student information to students dataframe. This generates a new student ID string\n    studentID=students.insert(newrow)\n    for app in applications:\n        # add the newly obtained student ID to the applications dictionaries\n        app[\'studentID\']=studentID[0]\n    # add the applications dictionaries to the applForm dataframe\n    applForm.insert(applications)\nstudents.df.head()')

students.df.shape

applForm.df.shape

students.save('collegedata_students.csv')
applForm.save('collegedata_applications.csv')

if ('students' in locals()): 
    students.cleanup()
    del students
if ('applForm' in locals()): del applForm
students=ti.Student()
applForm = ti.ApplForm()
students.read('collegedata_students.csv')
applForm.read('collegedata_applications.csv')
students.df.head()

applForm.df.head()

