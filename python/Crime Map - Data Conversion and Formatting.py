import pandas as pd
import os
import geocoder
import requests

for folder in os.listdir('./raw'):
    file_list = []
    for file in os.listdir('./raw/' + folder):
        if(file.endswith('.xls')):
            data = pd.read_excel('./raw/' + folder + '/' + file)
            file_list.append(data)
   
    df = pd.concat(file_list, ignore_index=True)
    
    columns = ['Incident Number','Date', 'Time', 'Police District','Offense 1',
          'Offense 2', 'Offense 3', 'Offense 4', 'Offense 5', 'Location']
    df.columns = columns
    
    df = df[df['Offense 1'].notnull()]
    df = df[df['Location'].notnull()]
    
    df.to_csv('./raw/' + folder + 'Complete.csv', sep=';', index = False)

def address_to_latlong(address):
    '''Returns the latitude and longitude for a given address'''
    address += ", Milwaukee, WI"
    location = geocoder.arcgis(address, session = session)
    
    if not location.latlng:
        print("Couldn't Parse: "  + address )
    else:
        print(f'Parsed: {address} - {location.latlng}')
    return location.latlng

for file in os.listdir('./raw/'):
    if(file.endswith('.csv')):
        df = pd.read_csv('./raw/'+ file, sep =';')
    
        zipped = zip(df['Offense 1'].items(), df['Offense 2'].items(), 
        df['Offense 3'].items(), df['Offense 4'].items(), df['Offense 5'].items())
        keyword = 'theft'
        
        for off_1, off_2, off_3, off_4, off_5 in zipped:
            if(not keyword in off_1[1].lower() and
              (pd.isnull(off_2[1]) or not keyword in off_2[1].lower()) and
              (pd.isnull(off_3[1]) or not keyword in off_3[1].lower()) and
              (pd.isnull(off_4[1]) or not keyword in off_4[1].lower()) and
              (pd.isnull(off_5[1]) or not keyword in off_5[1].lower())):
                df = df.drop(off_1[0])
 
        df['Address'] = df['Location'].copy()
            
        theft = df.copy()
        m_theft = df.copy()
        
        zipped = zip(df['Offense 1'].items(), df['Offense 2'].items(), 
        df['Offense 3'].items(), df['Offense 4'].items(), df['Offense 5'].items())
        keyword = 'motor vehicle theft'
        
        for off_1, off_2, off_3, off_4, off_5 in zipped:
            if(not keyword in off_1[1].lower() and
                (pd.isnull(off_2[1]) or not keyword in off_2[1].lower()) and
                (pd.isnull(off_3[1]) or not keyword in off_3[1].lower()) and
                (pd.isnull(off_4[1]) or not keyword in off_4[1].lower()) and
                (pd.isnull(off_5[1]) or not keyword in off_5[1].lower())):
                m_theft = m_theft.drop(off_1[0])
            else:
                theft = theft.drop(off_1[0])
                
        
        with requests.Session() as session:
            # This is likely to be very slow
            # depending on the geocoding service used
            theft['Location'] = theft['Location'].apply(address_to_latlong)
            m_theft['Location'] = m_theft['Location'].apply(address_to_latlong)
            
        theft.to_csv('./' + 'theft_' + file.split('Complete.csv')[0] + '.csv', sep=';', index = False)
        m_theft.to_csv('./' + 'm_theft_' + file.split('Complete.csv')[0] + '.csv', sep=';', index = False)
        print(f'Finished {file}')



