# data analysis and wrangling
import pandas as pd
import boto3
import numpy as np

already_imported = True

resource = boto3.resource(
    'dynamodb', endpoint_url='http://localhost:8000')

client = boto3.client(
    'dynamodb', endpoint_url='http://localhost:8000')

census_table = resource.Table('Census')
census_test_table = resource.Table('CensusTest')

column_names = [
    'age', 'workclass', 'fnlwgt', 
    'education', 'education-num', 'marital-status', 
    'occupation', 'relationship', 'race', 'sex', 
    'capital-gain', 'capital-loss', 'hours-per-week', 
    'native-country', 'salary']

train_df = pd.read_csv(
    'data/aws/census/adult.data', 
    header=None, names=column_names, 
    sep=', ', engine='python')

test_df = pd.read_csv(
    'data/aws/census/adult.test', 
    header=None, names=column_names, 
    sep=', ', engine='python', skiprows=1)

train_df.shape, test_df.shape

train_df.head()

train_df.info()

item_data_size = train_df[:1].squeeze().nbytes
attribute_names_size = sum([len(i) for i in column_names])
item_size = item_data_size + attribute_names_size

print(round(item_size/1024,2), "KB")

train_df['uid'] = train_df.index
test_df['uid'] = test_df.index

train_df.head()

if already_imported:
    print("Already imported data")
else:
    for index, row in train_df.iterrows():
        sample = row.squeeze().to_dict()

        for key, value in sample.items():
            if type(value) is np.int64:
                sample[key] = int(sample[key])

        census_table.put_item(Item=sample)
        print('Put {}'.format(index))

if already_imported:
    print("Already imported data")
else:
    for index, row in test_df.iterrows():
        sample = row.squeeze().to_dict()

        for key, value in sample.items():
            if type(value) is np.int64:
                sample[key] = int(sample[key])

        census_test_table.put_item(Item=sample)
        print('Put {}'.format(index))

table_description = client.describe_table(TableName='Census')
print(table_description)
table_description = client.describe_table(TableName='CensusTest')
print(table_description)

