import boto3
# Connect to simpleDB
client = boto3.client('sdb')
# Create a test domain (table in RDMS terms)
client.create_domain(
    DomainName='test')

# Create a test item and put attributes into it
# Note that values are always strings
test_item_name = 'U3G5603L'
item_attrs = [
    {'Name': 'Team', 'Value': 'U3G5603L'},
    {'Name': 'Channel', 'Value': 'U5LMN360', 'Replace': True},
    {'Name': 'ts', 'Value': '174435541.5', 'Replace': True}
    ]
response = client.put_attributes(
    DomainName='test',
    ItemName=test_item_name,
    Attributes=item_attrs
)

# Read back the attributes using consistent read, so that we don't get stale reads
response = client.get_attributes(
    DomainName='test',
    ItemName='U3G5603L',
    AttributeNames=[
        'ts',
        'Channel'
    ],
    ConsistentRead=True
)

response['Attributes']

# Delete test domain omain
response = client.delete_domain(
    DomainName='string'
)

import boto3
# Connect to simpleDB
client = boto3.client('sdb')

response = client.get_attributes(
    DomainName='awaybot',
    ItemName='T2BT8MVE3',
    AttributeNames=[
        'ts',
    ],
    ConsistentRead=True
)

response

import datetime
print(
    datetime.datetime.fromtimestamp(
        float(response['Attributes'][0]['Value'])
    ).strftime('%Y-%m-%d %H:%M:%S')
)

