get_ipython().magic('matplotlib inline')

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import psycopg2

pd.options.display.max_columns = 40

#Like before, we're going to select the relevant columns from the database:
connection = psycopg2.connect('dbname= threeoneone user=threeoneoneadmin password=threeoneoneadmin')
cursor = connection.cursor()

cursor.execute('''SELECT createddate, closeddate, borough FROM service;''')
data = cursor.fetchall()
data = pd.DataFrame(data)

data.columns = ['createddate','closeddate','borough']

data['cryear'] = [x.year for x in data['createddate']]
data['crmonth'] = [x.month for x in data['createddate']]

#filter out bad casesa
import datetime

today = datetime.date(2016,5,29)
janone = datetime.date(2010,1,1)

data = data.loc[(data['closeddate'] > data['createddate']) | (data['closeddate'].isnull() == True)]

databyyear = data.groupby(by='cryear').count()
databyyear

data['timedelta'] = data['closeddate'] - data['createddate']
data['timedeltaint'] = [x.days if pd.isnull(x) == False else None for x in data['timedelta']]

data.groupby(by='cryear').mean()

databyyear['propclosed'] = databyyear['closeddate']/databyyear['createddate']
databyyear

