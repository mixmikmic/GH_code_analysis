import csv
file = open("guns.csv",'r')
data = list(csv.reader(file))
print(data[:5])

headers = data[1:]
print(headers[:5])

year_count = {}
for item in headers:
    if item[1] in year_count:
        year_count[item[1]] += 1
    else:
        year_count[item[1]] = 1
    
print (year_count)

import datetime
dates = []
for item in headers:
    dates.append(datetime.datetime(year = int(item[1]), month = int(item[2]), day = 1))

date_counts = {}
for item in dates:
    if item in date_counts:
        date_counts[item] += 1
    else:
        date_counts[item] = 1

print(date_counts)

sex_count = {}
for item in headers:
    if item[5] in sex_count:
        sex_count[item[5]] += 1
    else:
        sex_count[item[5]] = 1
    
print (sex_count)

race_count = {}
for item in headers:
    if item[7] in race_count:
        race_count[item[7]] += 1
    else:
        race_count[item[7]] = 1
    
print (race_count)

file = open("census.csv",'r')
census = list(csv.reader(file))

new_census = census[1:]

race = {}
race["Asian/Pacific Islander"] = 15834141
race["Black"] = 40250635
race["Native American/Native Alaskan"] = 3739506
race["Hispanic"] = 44618105
race["White"] = 197318956

race_per_hundredk={}
for key,value in race_count.items():
    race_per_hundredk[key] = (race_count[key]/race[key])*100000

print(race_per_hundredk)

hom_count = {}
for item in headers:
    if item[7] in hom_count:
        if item[3]=="Homicide":
            hom_count[item[7]] += 1
    else:
        hom_count[item[7]] = 1
    
race_per_hundredk = {}
for key,value in hom_count.items():
    race_per_hundredk[key] = (hom_count[key]/race[key]) * 100000

print (race_per_hundredk)



