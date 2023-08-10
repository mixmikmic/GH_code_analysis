import os
import json
import numpy as np
import pandas as pd

def get_googlesheet_doc(gdocjson_pth, doc_name):
    import oauth2client
    import gspread
    
    # Get Google docs json token and scope for google sheets
    gdocstoken_json = os.path.join(gdocjson_pth, '.gdocs_Nanoos-fcdeeb760f83.json')
    scope = ['https://spreadsheets.google.com/feeds']

    # Retrieve credentials from JSON key of service account
    # oauth_ver = oauth2client.__version__
    try:
        from oauth2client.service_account import ServiceAccountCredentials
        credentials = ServiceAccountCredentials.from_json_keyfile_name(gdocstoken_json, scope)
    except:
        from oauth2client.client import SignedJwtAssertionCredentials
        with open(gdocstoken_json) as f:
            json_key = json.load(f)
        credentials = SignedJwtAssertionCredentials(json_key['client_email'],
                                                    json_key['private_key'], scope)
    gc = gspread.authorize(credentials)
    sheetgdoc = gc.open(doc_name)
    
    return sheetgdoc

import vizer.tsharvest.util as vhutil

vizer = vhutil.Vizer('nvs', False)

gdoc = get_googlesheet_doc(vizer.vizerspath, "sensor_configurations_mappings")

sheet = gdoc.worksheet('instruments')

sheetvalues = sheet.get_all_values()

# Convert data into pandas dataframe
df = pd.DataFrame(sheetvalues[1:], columns=sheetvalues[0])
df = df.convert_objects(convert_numeric=True)
df.head()

# Get Platforms
json_data = df[['platform', 'instrument', 'depth_m', 'mfn',
                'deployment', 'data_logger', 'subtype',
                'magnetic_declin_correction']].reset_index(drop=True)
platforms = json_data['platform'].unique()

# Create platform dictionary. Eliminate instruments with blank instrument strings,
# and platforms containing only such instruments.
platforms_dct = {}
for platform in platforms:
    instruments_df = json_data.loc[json_data['platform'] == platform]    
    instruments_tmp_dct = {}
    for idx, instruments_df_row in instruments_df.iterrows():
        row_dct = instruments_df_row.to_dict()
        instrument = row_dct['instrument']
        
        row_dct['mfn'] = True if row_dct['mfn'] == 'x' else False
        if row_dct['subtype'] != '':
            row_dct['subtype'] = int(row_dct['subtype'].split('::')[1])
        else:
            row_dct['subtype'] = None
        if np.isnan(row_dct['magnetic_declin_correction']):
            row_dct['magnetic_declin_correction'] = None
        
        row_dct.pop('platform', None)
        row_dct.pop('instrument', None)
        
        if len(instrument) > 1:
            instruments_tmp_dct[instrument] = row_dct
        
    if instruments_tmp_dct:
        platforms_dct[platform] = instruments_tmp_dct

platforms_dct.keys()

platforms_dct['CE09OSSM'].keys()

# prints the JSON structured dictionary
jsont_str = json.dumps(platforms_dct, sort_keys=True, indent=4)
print(jsont_str)

fpth = os.path.join(vizer.vizerspath, 'nvs', 'siso_ooi_harvest.json')
with open(fpth, 'w') as fojson:
    fojson.write(jsont_str)

