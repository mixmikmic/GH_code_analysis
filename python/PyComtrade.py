def comtrade(reporter,partner,year,freq,commoditycode):
    #Import your libraries
    import json
    import urllib
    import pandas as pd
    
    #Import the index of countries
    partnerurl = 'https://comtrade.un.org/data/cache/partnerAreas.json'
    areas = urllib.request.urlopen(partnerurl)
    data = json.loads(areas.read())
    data = data['results']
    index = {}
    for i in range(len(data)):
        upper = data[i]['text']
        lower = upper.lower()
        index.update({lower: data[i]['id']})
    
    #Retrieve numeric codes for reporter and partner
    reporter = index[str(reporter)]
    partner =  index[str(partner)]
    
    #Set the URL API
    url = 'http://comtrade.un.org/api/get?' +         'max=50000&' +         'type=C&' +         'freq=' + str(freq) + '&' +         'px=HS&' +         'ps=' + str(year) + '&' +         'r=' + reporter + '&' +         'p=' + partner + '&' +         'rg=all&' +         'cc=' + commoditycode + '&' +         'fmt=json'
    
    #Import data with the API, transform JSON into a frame
    urlopen = urllib.request.urlopen(url)
    data = json.loads(urlopen.read())
    data = pd.io.json.json_normalize(data['dataset'])
    
    #Return the data
    return data

#Import data from 2000 through 2014
first = 2000
last = 2014

reporter = 'brazil'
partner = 'usa'
freq = 'A'
ccode = 'TOTAL'

for year in list(range(first,last)):
    if year == first:
        frame = comtrade(reporter, partner, year, freq, ccode)
    else:
        framet = comtrade(reporter, partner, year, freq, ccode)
        frame = frame.append(framet)

print(frame)

#Print only the relevant information

print(frame[['yr','ptTitle','rtTitle','rgDesc','TradeValue']])

