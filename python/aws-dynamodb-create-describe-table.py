import boto3

resource = boto3.resource(
    'dynamodb',
    endpoint_url='http://localhost:8000')

client = boto3.client(
    'dynamodb',
    endpoint_url='http://localhost:8000')

table_to_create = 'Census'
print('Checking if table exists')

try:
    table_description = client.describe_table(TableName=table_to_create)
    print('Table {} already exists'.format(table_to_create))
    print('Table description:')
    print(table_description)

# Exception raised by describe_table if table does not exist
except Exception as e:
    if 'ResourceNotFoundException' in str(e):
        print('Table does not exist')
        print('Creating table')
        table = resource.create_table(
            TableName=table_to_create,
            KeySchema=[
                {'AttributeName': 'uid', 'KeyType': 'HASH'}
            ],
            AttributeDefinitions=[
                {'AttributeName': 'uid', 'AttributeType': 'N' }
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 10,
                'WriteCapacityUnits': 10
            }
        )
        
        #wait for contirmation that the table exists
        table.meta.client.get_waiter(
            'table_exists').wait(TableName=table_to_create)
        print('Table {} created'.format(table_to_create))
    else:
        print('Table cannot be created')
        raise

