import os, sys, email
import numpy as np 
import pandas as pd
from boto.s3.key import Key
import boto
import zipfile

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Read the data into a DataFrame
emails_df = pd.read_csv('../input/emails.csv')
print(emails_df.shape)
emails_df.head()

# A single message looks like this
print(emails_df['message'][0])

## Helper functions
def get_text_from_email(msg, max_word_len=30):
    '''To get the content from email objects'''
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            payload = part.get_payload()
            payload = ' '.join(filter(lambda x: len(x) < max_word_len,  payload.split()))
            parts.append( payload )
    return ''.join(parts)

def split_email_addresses(line):
    '''To separate multiple email addresses'''
    if line:
        addrs = line.split(',')
        addrs = frozenset(map(lambda x: x.strip(), addrs))
    else:
        addrs = None
    return addrs

# Parse the emails into a list email objects
messages = list(map(email.message_from_string, emails_df['message']))

# Get fields from parsed email objects
keys = messages[0].keys()
for key in keys:
    emails_df[key] = [doc[key] for doc in messages]
# Parse content from emails
emails_df['content'] = list(map(get_text_from_email, messages))
# Split multiple email addresses
emails_df['From'] = emails_df['From'].map(split_email_addresses)
emails_df['To'] = emails_df['To'].map(split_email_addresses)

# Extract the root of 'file' as 'user'
emails_df['user'] = emails_df['file'].map(lambda x:x.split('/')[0])

# cleanup
del messages
emails_df.drop('message', axis=1, inplace=True)

emails_df.head()

print('shape of the dataframe:', emails_df.shape)
# Find number of unique values in each columns
for col in emails_df.columns:
    print(col, emails_df["content"].nunique())
    
print("content length: {}".format(emails_df["content"].map(len).max()))

# Set index and drop columns with two few values
emails_df = emails_df.set_index('Message-ID')    .drop(['file', 'Mime-Version', 'Content-Type', 'Content-Transfer-Encoding'], axis=1)
# Parse datetime
emails_df['Date'] = pd.to_datetime(emails_df['Date'], infer_datetime_format=True)
emails_df.dtypes

def save_to_s3(file_name):
    s3 = boto.connect_s3()
    b = s3.get_bucket('brianray')
    k = Key(b)
    k.key = file_name
    k.set_contents_from_filename(file_name)
    k.set_acl('public-read')
    return k.generate_url(expires_in=0, query_auth=False)

def zipit(file_name):
    zip_file_name = "{}.zip".format(file_name)
    zf = zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED)
    try:
        zf.write(file_name)
    finally:
        zf.close()
    return zip_file_name

import glob
list_found = {}
cats = []
for path in glob.glob("enron_with_categories/*/*.txt"):
    batch, filename = path.split("/")[1:]
    contents = open(path, "r").read()

    try:
        email_parsed = email.message_from_string(contents)
        list_found[email_parsed['Message-ID']] = [x.split(',') 
                                                  for x in 
                                                  open(path.replace(".txt", ".cats")).read().split()]
    except Exception as e:
        print("error: {}".format(e))
        

for x in range(12):
    x += 1
    emails_df['Cat_{}_level_1'.format(x)] = None
    emails_df['Cat_{}_level_2'.format(x)] = None
    emails_df['Cat_{}_weight'.format(x)] = None    

emails_df.columns

emails_df['labeled'] = False       
for item, val in list_found.items():
    emails_df.loc[item, 'labeled'] = True
    i = 0
    for lev1, lev2, weight in val:
        i += 1
        emails_df.loc[item, 'Cat_{}_level_1'.format(i)] = lev1
        emails_df.loc[item, 'Cat_{}_level_2'.format(i)] = lev2
        emails_df.loc[item, 'Cat_{}_weight'.format(i)] = weight      

emails_df.columns

emails_df[emails_df['labeled'] == True]

len(emails_df.columns)

emails_df.reset_index(level=0, inplace=True)

emails_df.head()

filename = "enron_05_17_2015_with_labels_v2.csv"
emails_df.to_csv(filename)
save_to_s3(zipit(filename))



chunks = emails_df.groupby(np.arange(len(emails_df)) // 100000)
for i, chunk in chunks:
    name = "enron_05_17_2015_with_labels_v2_100K_chunk_{}_of_{}.csv".format(i+1, len(chunks))
    chunk.to_csv(name)
    print(save_to_s3(zipit(name)))

