#The dataset is first loaded into the guns.csv file
import csv
f=open('guns.csv','r')
data=list(csv.reader(f))


data[:5]

#Extract header
headers=data[0]
#remove header from list dats
data=data[1:]
headers

#counting how many deaths occured each year
year_counts={}
years=[]
for row in data:
    years.append(row[1])
for year in years:
    if year in year_counts:
        year_counts[year]+=1
    else:
        year_counts[year]=1
print(year_counts)

import datetime
dates=[]
for row in data:
    date=datetime.datetime(year=int(row[1]), month=int(row[2]), day=1)
    dates.append(date)
date_counts={}
for lst in dates:
    if lst in date_counts:
        date_counts[lst]+=1
    else:
        date_counts[lst]=1
print(date_counts)

#Counting male and female deaths
sexes=[row[5] for row in data]
sex_counts={}
#sex_counts.fromkeys(l,0)
for sex in sexes:
    if sex not in sex_counts:
        sex_counts[sex]=1
    else:
        sex_counts[sex]+=1
sex_counts

#Counting deaths by race
races=[row[7] for row in data]
race_counts={}
for race in races:
    if race not in race_counts:
        race_counts[race]=1
    else:
        race_counts[race]+=1
print(race_counts)

#Read census data from csv into a list
f=open("census.csv","r")
census=list(csv.reader(f))

#Explore census data
census

race_counts

mapping={"Asian/Pacific Islander":15159516+674625,
        "Black":40250635,
        "Native American/Native Alaskan":3739506,
        "Hispanic":44618105,
        "White":197318956}
race_per_hundredk={}

#Counting death per 100000 per race
for key,value in race_counts.items():
    if key in mapping:
       race_per_hundredk[key]=value/mapping[key]*100000
race_per_hundredk    

race_counts

homicide_race_counts={}
intents=[row[3] for row in data]
races=[row[7] for row in data]
for idx,race in enumerate(races):
    if intents[idx]=='Homicide':
        if race not in homicide_race_counts:
            homicide_race_counts[race]=1
        else:
            homicide_race_counts[race]+=1
homicide_race_counts

homicide_race_per_hundredk={}
for k,v in homicide_race_counts.items():
    if k in mapping:
        homicide_race_per_hundredk[k]=(v/mapping[k])*100000
homicide_race_per_hundredk



