# My imports
import json
import os
import gspread
from oauth2client.client import SignedJwtAssertionCredentials
import pandas as pd
import pydocumentdb.document_client as document_client
from pandas.io.json import read_json

# Specify my google drive api credentials
json_key = json.load(open('MessyDoc-8f814e3f2a78.json'))
scope = ['https://spreadsheets.google.com/feeds']
credentials = SignedJwtAssertionCredentials(json_key['client_email'], json_key['private_key'].encode(), scope)

# Using gspread module and my credentials, grab the google doc I want
gc = gspread.authorize(credentials)
wksheet = gc.open("SSF_Crop_Master_2012_Master_crop_master").worksheet('latest')

# Specify my DocumentDB settings
DOCUMENTDB_HOST = 'https://testingflask.documents.azure.com:443/'
DOCUMENTDB_KEY = 's610r3ylWxHNW8...=='
DOCDB_DATABASE = 'mladsapp'
DOCDB_COLLECTION_USER = 'user_collection'
DOCDB_COLLECTION_MASTER = 'master_collection'
DOCDB_MASTER_DOC = 'masterdoc'

# make a client connection
client = document_client.DocumentClient(DOCUMENTDB_HOST, {'masterKey': DOCUMENTDB_KEY})

# Read databases and get our working database
db = next((data for data in client.ReadDatabases() if data['id'] == DOCDB_DATABASE))

# Read collections and get the "master collection"
coll_master = next((coll for coll in client.ReadCollections(db['_self']) if coll['id'] == DOCDB_COLLECTION_MASTER))

# Read master document and place data into dataframe
master_doc = next((doc for doc in client.ReadDocuments(coll_master['_self']) if doc['id'] == DOCDB_MASTER_DOC))
raw_data_df = read_json(master_doc['data'])
raw_data_df.columns = read_json(master_doc['data_headers'])

print(raw_data_df.shape)

# Tidy up column names
cols = raw_data_df.columns
raw_data_df.columns = [e[0].encode('utf-8') for e in cols]

# Let's add a new column
#print(raw_data_df.columns)
a = raw_data_df['Seedingdate']
a = [e + '-2012' for e in a]
from datetime import datetime
t1 = datetime.strptime(a[0], '%d-%b-%Y')

b = raw_data_df['harvestdate'].iloc[:,0]
b = [e + '-2012' for e in b]
import time
t2 = datetime.strptime(b[0], '%d-%b-%Y')

days = (t2 - t1).days

# Add this column to data
raw_data_df['growingperiod_days'] = days

# make a client connection
client = document_client.DocumentClient(DOCUMENTDB_HOST, {'masterKey': DOCUMENTDB_KEY})

# Read databases and get our working database
db = next((data for data in client.ReadDatabases() if data['id'] == DOCDB_DATABASE))

# Read collections and get the "master collection"
coll_master = next((coll for coll in client.ReadCollections(db['_self']) if coll['id'] == DOCDB_COLLECTION_MASTER))

# Convert data values in df to json list of lists
values = raw_data_df.to_json(orient = 'values')

# Define a document definition
document_definition = { 'id': DOCDB_MASTER_DOC,
                       'timestamp': datetime.now().strftime('%c'),
                        'data': values,
                        'data_headers': pd.Series(raw_data_df.columns).to_json(orient = 'values')}

# Update the document in DocDB!
doc_updated = client.UpsertDocument(coll_master['_self'], document_definition)

# Some functions for updating (and concurrently) publishing a google spreadsheet doc
def numberToLetters(q):
    '''This converts a number,q,  into proper column name format for spreadsheet (e.g. R1C28 -> AB1).'''
    q = q - 1
    result = ''
    while q >= 0:
        remain = q % 26
        result = chr(remain+65) + result;
        q = q//26 - 1
    return result

def update_worksheet(wksheet, df):
    '''This function updates a given worksheet (wksheet)
    with the values in the dataframe (df).'''

    # TODO: confirm there are enough columns in existing doc to match query

    columns = df.columns.values.tolist()
    # selection of the range that will be updated
    cell_list = wksheet.range('A1:'+numberToLetters(len(columns))+'1')

    # modifying the values in the range
    for cell in cell_list:
        val = columns[cell.col-1]
        if type(val) is str:
            val = val.decode('utf-8')
        cell.value = val
    # update in batch
    wksheet.update_cells(cell_list)

    #number of lines and columns
    num_lines, num_columns = df.shape
    # selection of the range that will be updated
    cell_list = wksheet.range('A2:'+numberToLetters(num_columns)+str(num_lines+1))
    # modifying the values in the range
    for cell in cell_list:
        val = df.iloc[cell.row-2,cell.col-1]
        if type(val) is str:
            val = val.decode('utf-8')
        elif isinstance(val, (int, long, float, complex)):
            # note that we round all numbers
            val = int(round(val))
        cell.value = val
    # update in batch
    wksheet.update_cells(cell_list)

# Specify my DocumentDB settings
DOCUMENTDB_HOST = 'https://testingflask.documents.azure.com:443/'
DOCUMENTDB_KEY = 's610r3ylWxHNW87xKJYOmIzPWW/bHJNM7r4JCZ4PmSyJ2gUIEnasqH5wO9qkCY2LFkPV8kMulRa/U8+Ws9csoA=='
DOCDB_DATABASE = 'mladsapp'
DOCDB_COLLECTION_MASTER = 'master_collection'
DOCDB_MASTER_DOC = 'masterdoc'

# Again, specify my google drive api credentials
json_key = json.load(open('MessyDoc-8f814e3f2a78.json'))
scope = ['https://spreadsheets.google.com/feeds']
credentials = SignedJwtAssertionCredentials(json_key['client_email'], json_key['private_key'].encode(), scope)

# Using gspread module and my credentials, grab the google doc I want
gc = gspread.authorize(credentials)
wksheet = gc.open("SSF_Crop_Master_2012_Master_crop_master").worksheet('latest')

# make a client connection
client = document_client.DocumentClient(DOCUMENTDB_HOST, {'masterKey': DOCUMENTDB_KEY})

# Read databases and get our working database
db = next((data for data in client.ReadDatabases() if data['id'] == DOCDB_DATABASE))

# Read collections and get the "user collection"
coll_master = next((coll for coll in client.ReadCollections(db['_self']) if coll['id'] == DOCDB_COLLECTION_MASTER))

# Get master doc from DocDB and place into dataframe
master_doc = next((doc for doc in client.ReadDocuments(coll_master['_self']) if doc['id'] == DOCDB_MASTER_DOC))
master_data_df = read_json(master_doc['data'])
headers = read_json(master_doc['data_headers'])
master_data_df.columns = headers

# update all cells in master google doc with data in master doc from db
# this takes a minute or two (maybe put into a separate view function)
update_worksheet(wksheet, master_data_df)



