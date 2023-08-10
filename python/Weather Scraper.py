import time
import requests
import numpy as np
import pandas as pd

years = [y for y in range(2000, 2015)]
airports = ['KMSP']

for airport in airports: 
    # Each location gets its own DataFrame (and .csv)
    df = pd.DataFrame()   
    
    # Scrape through the years
    for year in years:
        url = 'http://www.wunderground.com/history/airport/%s/%i/1/1/CustomHistory.html?dayend=31&monthend=12&yearend=%i&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=&format=1' % (airport, year, year)
        dfNew = pd.read_csv(url)
        dfNew['location'] = airport
        df = pd.concat([df, dfNew])

        print airport, year
        time.sleep(5)
    
    # Post-Scrape DataFrame Processing
    df = df.set_index('CST').sort_index()
    
    # Write to .CSV
    csvPath = './data/weather-' + airport + '.csv'
    df.to_csv(csvPath)



