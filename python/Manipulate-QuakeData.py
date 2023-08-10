import pandas as pd
import math
import wget
import matplotlib.pyplot as plt
from matplotlib import animation

#PARSING PARAMETERS & CONTROL
    #init_year -> scan .cnss files starting with given year
    #end_year  -> scan .cnss files up to, but not including, this year (use 2017)
    #init_month-> scan .cnss files starting with given month
    #end_month -> scan .cnss files up to, but not including, this month (use 13)
    #location  -> scan .cnss files and parse out lines that contain the given location parameter, creates 'location'.csv
    #pToCSV    -> Boolean value, when set to False .cnss files will not be parsed into a csv
    #pAvgTable -> Boolean value, when set to False generated .csv files will not be averaged
    
init_year = 1990
end_year = 2017
init_month = 1
end_month = 13

p_toCSV = False    #Keep 'False' when testing functions, this will only work when 'eqData' directory exists and contains the .cnss files
p_avgTable = False #Keep 'False'
dl_csvData = False #Triggers to False once data has been downloaded, this prevents duplicate downloads

location = "CI"

#parseToCSV(directory,location)
#-This function is used to convert the .cnss raw data to a usable .csv format
#-The raw data set did not contain any hard set identifiers
#-Because of this the most reliable method of extracting information was to scan through individual characters per line
#-As of now, changing the formatting requires some modification of the code which is tedious

def parseToCSV(directory,location):
    fpath = str("eqData/" + directory)
    fo = open(fpath)

    newFile = open('mag_locationsCI.csv', 'a')
    
    with fo as f:  #load
        for line in f:
            if location not in line:
                continue
            else:
                date_year = ""
                date_month = ""
                date_day = ""
                mag_str = ""
                
                lat = ""
                lng = ""
                
                mag = 0
                pwr = 0
                ct = 0    
                
                #scrape year, month, date
                for y in range (5,9):
                    date_year+= line[y]
                for m in range (9,11):
                    date_month+= line[m]
                for d in range (11, 13):
                    date_day+= line[d]
                    
                #extract lat, lng
                for lt in range(24,33):
                    lat += line[lt]
                for lg in range(33,43):
                    lng += line[lg]
                    
                #extract magnitude    
                for mg in range (130,134):
                    mag_str+= line[mg]
                
                if mag_str[0] == ' ':
                    continue
                
                mag = float(mag_str)
                pwr = (10**(3*mag/2))
                
                if mag > 5.0: 
                    ct = 1
                else:
                    ct = 0                
                
                newFile.write(date_year + ',' + date_month + ',' + date_day + ',' + lat + ',' + lng + ',' + str(mag) + ',' + str(pwr) + ',' + str(ct) + '\n')
    
    fo.close()
    return

#Create .csv file with given range of year and months
if p_toCSV == True:
    for x in range(init_year, end_year):
        for y in range(init_month, end_month):
            if y<10:
                directory = str(x) + '.0' + str(y) + '.cnss'
            else:
                directory = str(x) + '.' + str(y) + '.cnss'
        
            parseToCSV(directory, location)    

#We used this code to creatte a combined dataset grouped by year, month and day.
#The remaining data was aggregated in two methods, by sum and mean

def manCSV(directory):
    #csv_path = str(directory + ".csv")
    csv_path = str('combined.csv')

    IDs = ['year', 'month', 'day', 'mag', 'pwr', 'ct']
    
    df = pd.read_csv(csv_path, names=IDs).groupby(['year', 'month', 'day'])
    
    
    grouped = pd.DataFrame(df.agg(['sum', 'mean']).join(pd.DataFrame(df.size(), 
                                columns=['counts'])))
    #print(grouped)
    grouped.to_csv("m_" + csv_path, sep=',')
    
   
    return grouped







