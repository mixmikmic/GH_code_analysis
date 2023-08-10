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
RAW_PICK = os.path.join(PROJ_ROOT, "data", "raw", "harvard_sports.p")

sports = pd.read_pickle(RAW_PICK)

sports.columns

len(sports)

sports.rosters

# The top entry for every table describes what metadata is available for players in that table.
sports.iloc[0]['rosters']['2015-16'][:1]

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


fish = getFieldnames(sports.iloc[32]['rosters']['2008-09'])
fish

sports.iloc[32]['rosters']['2008-09'][1]

for i, val in enumerate(fish):
    print val

# inexplicably, the headshot column sneaks into the table even though t
# the table header doesn't have a label for headshot!

# beware data issues: someone's weight entered as "20-0"
# http://www.gocrimson.com/sports/mbkb/1999-00/roster

rosters = []

for i, row in sports.iterrows(): # for each sport
    sportName = row['sportCode']
    
    for season, roster in row['rosters'].iteritems():    # check for each season
        nRoster = {}
        players = [] # new array of players for every season
        
        fieldNames = getFieldnames(roster)
        lenField = len(fieldNames)
           
        nRoster['sport'] = sportName             # this is the foreign key, when combined with the season
        nRoster['season'] = season
        nRoster['metadata'] = fieldNames
        
        # assumes that fieldnames are constant over all years for a given
        # sport.
        for person in roster[1:]: # skip over header row
            lenPerson = len(person)
            # iterate thru the tail rows of the roster, 
            nPlayer = {}
            nPlayer['sport'] = sportName
            nPlayer['season'] = season

            ## for each fieldname for the person
            for i, val in enumerate(fieldNames):
                if (lenField < lenPerson):
                    # for any sport with more cells than # of header labels
                    #, that means that a blank "picture" cell has been included.
                    #, and shoudl be skipped
                    fieldVal = person[i+1] # skip portrait
                else:
                    fieldVal = person[i]
    
                
                if type(fieldVal) is dict:
                    nPlayer[val + '_link'] = fieldVal['link']
                    nPlayer[val] = fieldVal['name']
                else:
                    nPlayer[val] = fieldVal
                    
            players.append(nPlayer)
        
        nRoster['players'] = players
        
        rosters.append(nRoster)
        

rosters[0]['players'][:2]

len(rosters)

players = []

for roster in rosters:
    players.extend(roster['players'])

len(players)

pdf = pd.DataFrame(players)

print pdf.columns
pdf.shape

pdf.head()

# check which players are doing stuff every semester
pdf.groupby('Name').count().sort_values('sport', ascending=False).head(10)

def lookupPlayer(name):
    return pdf[pdf['Name'] == name]

# Who is a multi sport athlete?
pdf.groupby('Name')['sport'].nunique().sort_values(ascending=False).head(10)

# how many of these people are there?!
multis = pdf.groupby('Name')['sport'].nunique().sort_values(ascending=False)

# there were 152 multisport athletes! Their names are
multis = multis[multis >= 2]

multis.shape

multis.index

# we need to make a separate table for when each of these people started yale so you can ask if there's a pattern to this!

lookupPlayer("James Leakos")

lookupPlayer("Connor Green")

# We know his height from one sport
# he has different bios in different sports.
# this table is actually a "relational" table called "played". Every time someone plays in a season, they get an entry here.

# note that if the play every year, their bio might change from year to year. in some cases their major is added in their oldest bio.

lookupPlayer("Jason Michas")

def splitCityReg(string):
    if string == "-":
        return ["-", "-"]
    else:
        
        pair = string.split(", ")
        if len(pair) is 2:
            return pair
        else:
            return [string,'NOREGION?']

pdf['City'], pdf['Region'] = zip(*pdf['Hometown'].apply(lambda x: splitCityReg(x) ))

m_cities_regs = pdf.groupby('Hometown')['Name'].nunique().sort_values(ascending=False)
m_cities_regs

## By school, exeter is at the top once again
## these queries can be rewritten using sql

m_school = pdf.groupby('High School')['Name'].nunique().sort_values(ascending=False)
m_school

pdf['Hometown'][0].split(",")

m_city = pdf.groupby('City')['Name'].nunique().sort_values(ascending=False)
m_city

# note the misspelling of sweeden (sic) and the doubling 
# of the number of people from california

m_reg = pdf.groupby('Region')['Name'].nunique().sort_values(ascending=False)
m_reg

pdf[pdf['Region'] == "NOREGION?"].head(20)

# if you're in canada, you get your country and your province
# sometimes people just leave the state off.

# this data should probably be cleaned by hand!
# people from big cities tend to leave the state off

EXPORT_PLAYERS = os.path.join(PROJ_ROOT, 'data', 'interim', 'harvard_players.json')
EXPORT_ROSTERS = os.path.join(PROJ_ROOT, 'data', 'interim', 'harvard_rosters.p')

pdf.to_json(EXPORT_PLAYERS)

rdf = pd.DataFrame(rosters)

rdf.shape

# verify later that tables are the same across all years for any given sport.
rdf.head()

# This is going to be somewhat space inefficient for now, but it's OK
rdf.to_pickle(EXPORT_ROSTERS)

