import boto3
import pandas as pd

session = boto3.Session()
client = session.client('s3')

result =  client.list_objects(Bucket = 'usdot-its-cvpilot-public-data', Delimiter='/', Prefix='wydot/BSM/20170815T234600')

print(len(result.get('CommonPrefixes')))

if result.get('CommonPrefixes') is not None:
    for o in result.get('CommonPrefixes'):
        print ('subfolder : ', o.get('Prefix'))
else:
    print('No folder found for that prefix')

def dir_keys(client, bucket, prefix='', filekeys=[]):
    """
    Lists all file keys from a given prefix in an S3 bucket.  If no prefix is given all file keys are returned

    :param client: S3 connection object
    :param bucket: Name of bucket to search
    :param prefix: Prefix for a given folder
    :param filekeys: list for filekeys
    :return: updated filekey list with added files from search
    """
    paginator = client.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=prefix):
        if result.get('Contents') is not None:
            for file in result.get('Contents'):
                if file.get('Key') != 'unknownDataType':
                    filekeys.append(file.get('Key'))
        if result.get('CommonPrefixes') is not None:
            for subdir in result.get('CommonPrefixes'):
                dir_keys(client, bucket, subdir.get('Prefix'), filekeys)
    return filekeys

filekeys = dir_keys(client, 'usdot-its-cvpilot-public-data', 'wydot/BSM/20170815T234600')
print('Total number of files:', str(len(filekeys)))

# Create local directory
import os
cwd = os.getcwd()
local_directory = cwd + os.sep + 'tmp' + os.sep
if not os.path.exists(local_directory):
    os.makedirs(local_directory)

# Download Files
for file in filekeys:
    client.download_file('usdot-its-cvpilot-public-data', file, local_directory + file.split('/')[-1])
print('{} Files loaded to {}'.format(str(len(os.listdir(local_directory))), '/tmp/'))

get_ipython().system('cat ./tmp/wydot-filtered-bsm-1502840971677.json')

import pprint
import json
data = json.loads(open("./tmp/wydot-filtered-bsm-1502840973991.json").read())
pprint.pprint(data)

import glob

read_files = glob.glob(local_directory + "*.json")
with open(local_directory + "merged_file.json", "w") as outfile:
    data = []
    for f in read_files:
        data.append(open(f, "r").read())
    outfile.write("[" + ','.join(data[1:]) + "]")

from pandas.io.json import json_normalize

file_json = json.load(open(local_directory + "merged_file.json","r"))

# for element in file_json: 
#     del element['partII'] 

result = json_normalize(data=file_json, meta=['metadata', ['payload', 'data']])
result.head()

result['payload.data.coreData.speed'].describe()

result['metadata.generatedAt']= result['metadata.generatedAt'].str[:-5]
result['metadata.generatedAt'] = pd.to_datetime(result['metadata.generatedAt'])
result['metadata.generatedAt']

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(result['metadata.generatedAt'], result['payload.data.coreData.speed'])



