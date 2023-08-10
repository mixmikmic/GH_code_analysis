# Google Authentication Libraries
import oauth2client, gspread
import json

# oauth2client version check and gspread
oauth_ver = oauth2client.__version__
gspread_ver = gspread.__version__

print "oauth2client version : {}".format(oauth_ver) 
print "gspread version : {}".format(gspread_ver)

if oauth_ver < "2.0.2":
    from oauth2client.client import SignedJwtAssertionCredentials

    json_key = json.load(open('XXXX.json'))
    # Get scope for google sheets
    # Gather all spreadsheets shared with the client_email: XXXX@appspot.gserviceaccount.com
    scope = ['https://spreadsheets.google.com/feeds']
    
    # Retrieve credentials from JSON key of service account
    credentials = SignedJwtAssertionCredentials(json_key['client_email'], json_key['private_key'], scope)
    
    # Authorize gspread to connect to google sheets
    gc = gspread.authorize(credentials)
else:
    from oauth2client.service_account import ServiceAccountCredentials
    # Get scope for google sheets
    # Gather all spreadsheets shared with the client_email: XXXX@appspot.gserviceaccount.com
    scope = ['https://spreadsheets.google.com/feeds']

    # Retrieve credentials from JSON key of service account
    credentials = ServiceAccountCredentials.from_json_keyfile_name('XXXX.json', scope)

    # Authorize gspread to connect to google sheets
    gc = gspread.authorize(credentials)

# Get all spreadsheets available for NANOOS
gsheets = gc.openall()
# Get title of the spreadsheets
for i in range(0,len(gsheets)):
    print "{0} {1}".format(i,gsheets[i].title)

# Open sensor_configurations_mappings only
sc = gc.open("sensor_configurations_mappings")

# Get all worksheets in a sheet
wks = sc.worksheets()
wks

s1 = sc.get_worksheet(0)
s2 = sc.get_worksheet(1)
print s1, s2

# Import pandas and numpy to make data easier to view
get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
print "pandas version: {}".format(pd.__version__)
print "numpy version: {}".format(np.__version__)

# Getting all the values of sheet1
array1 = s1.get_all_values()
array2 = s2.get_all_values()

# Convert data into pandas dataframe
df = pd.DataFrame(array1)
df.columns = array1[0]
df.drop(df.index[0], inplace=True)
df = df.convert_objects(convert_numeric=True)
df.head()

# Convert data into pandas dataframe
df1 = pd.DataFrame(array2)
df1.columns = array2[0]
df1.drop(df1.index[0], inplace=True)
df1 = df1.convert_objects(convert_numeric=True)
df1.head()

def createJSON(df):
    # Get Platforms
    json_data = df[['platform','instrument','depth_m','mfn','deployment','data_logger','subtype']].reset_index(drop=True)
    platforms = json_data['platform'].unique()
    mainkey = dict()
    prop = dict()
    
    # Gather Platform info together
    plat = [json_data.loc[json_data['platform'] == p] for p in platforms]
    
    # Create JSON
    for i in range(0, len(plat)):
        instrum = dict()
        mainkey = dict()
        for j in range(0, len(plat[i]['platform'].values)):
            platform_name = plat[i]['platform'].values[j]
            instrument_name = plat[i]['instrument'].values[j]
            depth_m = plat[i]['depth_m'].values[j]
            mfn = plat[i]['mfn'].values[j]
            deployment = plat[i]['deployment'].values[j]
            data_logger = plat[i]['data_logger'].values[j]
            subtype = plat[i]['subtype'].values[j]

            # Check for mfn
            if mfn != '':
                mfn = True
            else:
                mfn = False
            # Getting subtype
            if subtype != '':
                subtype = subtype.split('::')[1]
            else:
                subtype = None

            prop['depth_m'] = float(depth_m)
            prop['mfn'] = mfn
            prop['deployment'] = deployment
            prop['data_logger'] = data_logger
            prop['subtype'] = subtype
            instrum['{}'.format(instrument_name)] = prop
            mainkey['{}'.format(platform_name)] = instrum
            prop = dict()
            
        # prints the JSON structured dictionary
        print json.dumps(mainkey, sort_keys=True, indent=4, separators=(',', ': '))
        # Output to JSON file 
        fj = open("{}.json".format(platform_name), 'w')
        fj.write(json.dumps(mainkey, sort_keys=False, indent=4, separators=(',', ': ')))
        fj.close()
createJSON(df)



