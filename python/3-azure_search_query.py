import requests
import json
import os
import csv
import pyexcel as pe
import codecs
import pandas as pd

# This is the service you've already created in Azure Portal
serviceName = 'your_azure_search_service_name'

# This is the index you've already created in Azure Portal or via the azsearch_mgmt.py script
indexName = 'your_index_name_to_use'

# Set your service API key, either via an environment variable or enter it below
#apiKey = os.getenv('SEARCH_KEY_DEV', '')
apiKey = 'your_azure_search_service_api_key'
apiVersion = '2016-09-01'

# Retrieval options to alter the query results
SEARCHFIELDS = None                            # use all searchable fields for retrieval
#SEARCHFIELDS = 'Keywords, SubsectionText'     # use selected fields only for retrieval
FUZZY = False                                  # enable fuzzy search (check API for details)
NTOP  = 5                                      # uumber of results to return

def getServiceUrl():
    return 'https://' + serviceName + '.search.windows.net'

def getMethod(servicePath):
    headers = {'Content-type': 'application/json', 'api-key': apiKey}
    r = requests.get(getServiceUrl() + servicePath, headers=headers)
    #print(r, r.text)
    return r

def postMethod(servicePath, body):
    headers = {'Content-type': 'application/json', 'api-key': apiKey}
    r = requests.post(getServiceUrl() + servicePath, headers=headers, data=body)
    #print(r, r.text)
    return r

def submitQuery(query, fields=None, ntop=10, fuzzy=False):
    servicePath = '/indexes/' + indexName + '/docs?api-version=%s&search=%s&$top=%d' %         (apiVersion, query, ntop)
    if fields != None:
        servicePath += '&searchFields=%s' % fields
    if fuzzy:
        servicePath += '&queryType=full'
    
    # Submit GET request
    r = getMethod(servicePath)
    if r.status_code != 200:
        print('Failed to retrieve search results')
        print(r, r.text)
        return
    
    # Parse and report search results
    docs = json.loads(r.text)['value']
    print('Number of search results = %d\n' % len(docs))
    for i, doc in enumerate(docs):
        print('Results# %d' % (i+1))
        print('Chapter title   : %s' % doc['ChapterTitle'].encode('utf8'))
        print('Section title   : %s' % doc['SectionTitle'].encode('utf8'))
        print('Subsection title: %s' % doc['SubsectionTitle'].encode('utf8'))
        print('%s\n' % doc['SubsectionText'].encode('utf8'))
        
def submitBatchQuery(query, fields=None, ntop=10, fuzzy=False):
    servicePath = '/indexes/' + indexName + '/docs?api-version=%s&search=%s&$top=%d' %         (apiVersion, query, ntop)
    if fields != None:
        servicePath += '&searchFields=%s' % fields
    if fuzzy:
        servicePath += '&queryType=full'

    # Submit GET request
    r = getMethod(servicePath)
    if r.status_code != 200:
        print('Failed to retrieve search results')
        print(query, r, r.text)
        return {}

    # Return search results
    docs = json.loads(r.text)['value']
    return docs

query = 'what is the tax bracket for married couple filing separately'
if query != '':
    # Submit query to Azure Search and retrieve results
    searchFields = SEARCHFIELDS
    submitQuery(query, fields=searchFields, ntop=NTOP)

# Input file coontaining the list of queries [tab-separated .txt or .tsv, Excel .xls or .xlsx]
infile  = os.path.join(os.getcwd(), '../sample/sample_queries.txt')
outfile = os.path.join(os.getcwd(), '../sample/sample_query_answers.xlsx')

if infile.endswith('.tsv') or infile.endswith('.txt'):
    records = pd.read_csv(infile, sep='\t', header=0, encoding='utf-8')
    rows = records.iterrows()
elif infile.endswith('.xls') or infile.endswith('.xlsx'):
    records = pe.iget_records(file_name=infile)
    rows = enumerate(records)
else:
    print('Unsupported query file extension. Options: tsv, txt, xls, xlsx')

# Dataframe to keep index of crawled pages
df = pd.DataFrame(columns = ['Qid', 'Query', 'Rank', 'SubsectionText', 'ChapterTitle', 'SectionTitle', 'SubsectionTitle', 'Keywords'])
        
for i, row in rows:
    qid   = int(row['Qid'])
    query = row['Query']
    # Submit query to Azure Search and retrieve results
    searchFields = SEARCHFIELDS
    docs = submitBatchQuery(query, fields=searchFields, ntop=NTOP, fuzzy=FUZZY)
    print('QID: %4d\tNumber of results: %d' % (qid, len(docs)))
    for id, doc in enumerate(docs):
        chapter_title    = doc['ChapterTitle']
        section_title    = doc['SectionTitle']
        subsection_title = doc['SubsectionTitle']
        subsection_text  = doc['SubsectionText']
        keywords         = doc['Keywords']

        df = df.append({'Qid'             : qid, 
                        'Query'           : query, 
                        'Rank'            : (id + 1), 
                        'SubsectionText'  : subsection_text,
                        'ChapterTitle'    : chapter_title,
                        'SectionTitle'    : section_title,
                        'SubsectionTitle' : subsection_title,
                        'Keywords'   : keywords},
                        ignore_index=True)

# Save all answers
df['Qid']  = df['Qid'].astype(int)
df['Rank'] = df['Rank'].astype(int)

if outfile.endswith('.xls') or outfile.endswith('.xlsx'):
    df.to_excel(outfile, index=False, encoding='utf-8')    
else:    # default tab-separated file
    df.to_csv(outfile, sep='\t', index=False, encoding='utf-8') 
print('Search results saved in file %s' % os.path.basename(outfile))



