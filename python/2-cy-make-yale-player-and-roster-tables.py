# Use w/ Anaconda Distribution
# For data manipulation
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
PROJ_ROOT = os.pardir

# For Scraping
from bs4 import BeautifulSoup
import requests
import time

# For munging
import re
import json

# Backoff time for large scrapes
THROTTLE_TIME = .05

# ipython magics
get_ipython().magic('load_ext watermark')
get_ipython().magic('matplotlib inline')

import missingno as msno

get_ipython().magic('watermark -a "Cameron Yick" -d -t -v -p pandas,seaborn,matplotlib,bs4 -g')

RAW_PATH = os.path.join(PROJ_ROOT, "data", "raw", "sports.json")
RAW_PICK = os.path.join(PROJ_ROOT, "data", "raw", "sports.p")

sports = pd.read_pickle(RAW_PICK)

sports.columns

sports.rosters

# The top entry for every table describes what metadata is available for players in that table.
sports.iloc[0]['rosters']['2016-17'][-1:]

def getFieldnames(rows):
    'Given an array of rows from a table, return a list of what to name all the keys in that table.'
    header = rows[0]
    names = []
    for fieldName in header:
        if type(fieldName) is dict:
            names.append(fieldName['name'])
        else:
            names.append(fieldName)
            
    return names

rosters = []

for i, row in sports.iterrows(): # for each sport
    sportName = row['name']
    
    for season, roster in row['rosters'].iteritems():    # check for each season
        nRoster = {}
        players = [] # new array of players for every season
        
        fieldNames = getFieldnames(roster)
           
        nRoster['sport'] = sportName             # this is the foreign key, when combined with the season
        nRoster['season'] = season
        nRoster['metadata'] = fieldNames
       
        #         print "{0} {1} {2}".format(season, sportName, len(roster[1:]))
        
        for person in roster[1:]: # iterate throw the tail rows of the roster, and drop the last entry
            nPlayer = {}
            nPlayer['sport'] = sportName
            nPlayer['season'] = season

            for i, val in enumerate(fieldNames):
                
                fieldVal = person[i+1] # do to skip the headshot column
                
                if type(fieldVal) is dict:
                    nPlayer[val + '_link'] = fieldVal['link']
                    nPlayer[val] = fieldVal['name']
                else:
                    nPlayer[val] = fieldVal
                    
            players.append(nPlayer)
        
        nRoster['players'] = players
        
        rosters.append(nRoster)
        

len(rosters)

players = []

for roster in rosters:
    players.extend(roster['players'])

len(players)

pdf = pd.DataFrame(players)

print pdf.columns
pdf.shape

pdf.head()

# beware inconsistency in high school name
pdf[pdf['Name'] == 'Emily Waligurski']

# check which players are doing stuff every semester
pdf.groupby('Name').count().sort_values('Cl.', ascending=False).head(10)

def lookupPlayer(name):
    return pdf[pdf['Name'] == name]

# Who is a multi sport athlete?
pdf.groupby('Name')['sport'].nunique().sort_values(ascending=False).head(10)

# how many of these people are there?!
multis = pdf.groupby('Name')['sport'].nunique().sort_values(ascending=False)

# there were 246 multisport athletes! Their names can be 
multis = multis[multis >= 2]

multis.index

# we need to make a separate table for when each of these people started yale so you can ask if there's a pattern to this!

lookupPlayer("Lindsey Raymond")

# We know his height from one sport
# he has different bios in different sports.
# this table is actually a "relational" table called "played". Every time someone plays in a season, they get an entry here.

# note that if the play every year, their bio might change from year to year. in some cases their major is added in their oldest bio.

lookupPlayer("Jacque Sahlberg")

pdf['Hometown/Region'], pdf['High School'] = zip(*pdf['Hometown/High School'].apply(lambda x: x.split(" / ")))

def splitCityReg(string):
    if string == "-":
        return ["-", "-"]
    else:
        
        pair = string.split(", ")
        if len(pair) is 2:
            return pair
        else:
            return [string,'NOREGION?']

pdf['City'], pdf['Region'] = zip(*pdf['Hometown/Region'].apply(lambda x: splitCityReg(x) ))

m_cities_regs = pdf.groupby('Hometown/Region')['Name'].nunique().sort_values(ascending=False)
m_cities_regs

m_school = pdf.groupby('High School')['Name'].nunique().sort_values(ascending=False)
m_school

pdf['Hometown/Region'][0].split(",")

m_city = pdf.groupby('City')['Name'].nunique().sort_values(ascending=False)
m_city

m_reg = pdf.groupby('Region')['Name'].nunique().sort_values(ascending=False)
m_reg

pdf[pdf['Region'] == "NOREGION?"].head(20)

# if you're in canada, you get your country and your province
# sometimes people just leave the state off.

# this data should probably be cleaned by hand!
# people from big cities tend to leave the state off

EXPORT_PLAYERS = os.path.join(PROJ_ROOT, 'data', 'interim', 'players.json')
EXPORT_ROSTERS = os.path.join(PROJ_ROOT, 'data', 'interim', 'rosters.p')

pdf.to_json(EXPORT_PLAYERS)

rdf = pd.DataFrame(rosters)

# verify later that tables are the same across all years for any given sport.
rdf.shape
rdf.head()

# This is going to be somewhat space inefficient for now, but it's OK
rdf.to_pickle(EXPORT_ROSTERS)

