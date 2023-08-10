import datadotworld as dw
import csv
import pandas as pd

# Datasets are referenced by their path
dataset_key = 'rflprr/d-4-d-hack-financial-disclosures'

# Or simply by their URL
dataset_key = 'https://data.world/rflprr/d-4-d-hack-financial-disclosures'

# Load dataset (onto the local file system)
dataset_local = dw.load_dataset(dataset_key, force_update=True)  # cached under ~/.dw/cache

# See what is in it
dataset_local.describe()

# create an empty array to hold the new, reshaped data

d = []

c = 0
for i in dataset_local.tables['filer-s-employment-agreements-and-arrangements']:
    file = i['file']
    disclosure_type = "employment"
    org = i['employer-or-party']
    location = i['city-state']
    c = c + 1
    d.append({'file': file, 'disclosure_type': disclosure_type, 'org': org, 'location': location})

print(str(c) + " records added from the employment table")
print(str(len(d)) + " records in the new dataset so far")

c = 0
for i in dataset_local.tables['filer-s-employment-assets-_-income-and-retirement-accounts']:
    file = i['file']
    disclosure_type = "assets"
    org = i['description']
    location = None
    c = c + 1
    d.append({'file': file, 'disclosure_type': disclosure_type, 'org': org, 'location': location})


print(str(c) + " records added from the assets table")
print(str(len(d)) + " records in the new dataset so far")

c = 0
for i in dataset_local.tables['filer-s-positions-held-outside-united-states-government']:
    file = i['file']
    disclosure_type = "positions"
    org = i['organization-name']
    location = i['city-state']
    c = c + 1
    d.append({'file': file, 'disclosure_type': disclosure_type, 'org': org, 'location': location})

print(str(c) + " records added from the positions table")
print(str(len(d)) + " records in the new dataset so far")

c = 0
for i in dataset_local.tables['filer-s-sources-of-compensation-exceeding-_5-000-in-a-year']:
    file = i['file']
    disclosure_type = "compensations"
    org = i['source-name']
    location = i['city-state']
    c = c + 1
    d.append({'file': file, 'disclosure_type': disclosure_type, 'org': org, 'location': location})
    
print(str(c) + " records added from the compensations table")
print(str(len(d)) + " records in the new dataset so far")

c = 0
for i in dataset_local.tables['liabilities']:
    file = i['file']
    disclosure_type = "liabilities"
    org = i['creditor-name']
    location = None
    c = c + 1
    d.append({'file': file, 'disclosure_type': disclosure_type, 'org': org, 'location': location})

print(str(c) + " records added from the liabilities table")
print(str(len(d)) + " records in the new dataset so far")

c = 0
for i in dataset_local.tables['other-assets-and-income']:
    file = i['file']
    disclosure_type = "other_assets"
    org = i['description']
    location = None
    c = c + 1
    d.append({'file': file, 'disclosure_type': disclosure_type, 'org': org, 'location': location})

print(str(c) + " records added from the other_assets table")
print(str(len(d)) + " records in the new dataset so far")

# let's preview a record to see how it looks

d[0]

for i in d:
    for o in dataset_local.tables['org_crosswalk']:
        if i['org'] == o['source_org_value']:
            i['org_cluster'] = o['org_group']

df = pd.DataFrame(d)

df.describe()

# to write the new dataset to csv, run this command: 
df.to_csv(path_or_buf="./d.csv")









