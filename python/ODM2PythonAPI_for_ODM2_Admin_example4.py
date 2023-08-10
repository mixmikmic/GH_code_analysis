get_ipython().run_line_magic('matplotlib', 'inline')

import sys
import os
import pprint
import numpy
import pandas as pd
import getpass
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.dates import DateFormatter
from IPython.display import display, HTML
import ipywidgets as widgets

from odm2api.ODMconnection import dbconnection
import odm2api.ODM2.services.readService as odm2
from odm2api.ODM2.models import *

#print("Enter your ODM2 username") 
container = widgets.Box() # would be nice If I could get a container to hold the 
# user name and password prompt, getpass doesn't seem to play well with the other 
# widgets though
username_text = widgets.Text(
    value='demo', placeholder='Enter username',
    description='', disabled=False)
username_output_text = widgets.Text(
    value='', placeholder='Enter username',
    description='Username',disabled=False)
database_address_text = widgets.Text(
    value='40.85.180.138', placeholder='Enter database address',
    description='',disabled=False)
database_address_output_text = widgets.Text(
    value='',placeholder='Enter database address',
    description='database address',disabled=False)
database_text = widgets.Text(
    value='odm2sandbox', placeholder='Enter database name',
    description='', disabled=False)
database_output_text = widgets.Text(
    value='', placeholder='Enter database name',
    description='database name', disabled=False)
def bind_username_to_output(sender):
    username_output_text.value = username_text.value
def bind_database_address_to_output(sender):
    database_address_output_text.value = database_address_text.value
def bind_database_to_output(sender):
    database_output_text.value = database_text.value     
    
def login(sender):
    #print('Database address : %s, Username: %s, database name: %s' % (
    #    database_address_text.value, username_text.value, database_text.value))
    container.close()    
    
username_text.on_submit(bind_username_to_output)
login_btn = widgets.Button(description="Login")
login_btn.on_click(login)
container.children = [username_text,database_address_text, database_text, login_btn]
container

print("enter your password: ")
p = getpass.getpass()

session_factory = dbconnection.createConnection('postgresql', database_address_text.value, database_text.value, 
                                                username_text.value, p) 

read = odm2.ReadODM2(session_factory)

#featureaction = 1700
results = read.getResults(actionid=30)
resultids = []
resultnames = {}

def print_result_name(result):
    print result
def on_change(change):
    print(change['new'])
    
for r in results:
    #print(r.ResultID)
    resultids.append(str(r.ResultID))
    detailr = read.getDetailedResultInfo(resultTypeCV = 'Time series coverage',resultID=r.ResultID)
    for detail in detailr:
        namestr = str(detail.SamplingFeatureCode + "- " + detail.MethodCode + "- "+ detail.VariableCode + "- " + detail.UnitsName)
        
        resultnames[namestr]=  detail.ResultID
    #print(detailr.Methods)
print(resultids)
resultWidget = widgets.Dropdown(options=resultnames)
rwidget = widgets.interactive(print_result_name,result=resultWidget)
rwidget.observe(on_change)
display(rwidget)

print(resultWidget.value)
ids = [resultWidget.value]
selectedResult = read.getDetailedResultInfo(resultTypeCV = 'Time series coverage',resultID=resultWidget.value)
SUNAResultValues = read.getResultValues(resultids=ids, starttime='2016-8-1', 
                                                 endtime= '2016-8-30')

print(SUNAResultValues.head())
#
SUNAResultValues.set_index('valuedatetime', inplace=True)
print(SUNAResultValues.index)
print(SUNAResultValues.index.name)

dateFmt = DateFormatter('%m-%d')
fig, ax = plt.subplots()
ax.xaxis_date()
#SUNAResultValues["valuedatetimed"] = pd.to_datetime(SUNAResultValues["valuedatetime"])
#SUNAResultValues["month-day"] = SUNAResultValues['valuedatetimed'].dt.strftime('%m-%d %H:%M') 
#print(SUNAResultValues["month-day"])
#mpl.dates.date2num(df.index.to_pydatetime())
#SUNAResultValues["valuedatetimed"]

ax.plot_date(  mpl.dates.date2num(SUNAResultValues.index.to_pydatetime()), SUNAResultValues['datavalue'])
ax.xaxis.set_major_formatter(dateFmt)

plt.xticks(rotation=70)
for result in selectedResult:
    plt.title(str(result.VariableCode) + ' September 2016')
    plt.ylabel(result.UnitsName)

SUNAResultValues['datavalue'].diff().hist(alpha=1, bins=15)
SUNAResultValues['datavalue'].describe()



