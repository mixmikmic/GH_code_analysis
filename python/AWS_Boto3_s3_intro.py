import boto3

# check that boto3 is able to pick up valid credentials
# Will print a list of profiles
session = boto3.Session(profile_name='d4d_tutorial')
session.available_profiles

# Tell boto3 which resource you will use
s3 = session.resource('s3')

# Specify AWS bucket
bucket = s3.Bucket('public-test-bucket-d4d')

# print all objects in bucket
for obj in bucket.objects.all():
    print(obj)

# get object by name
file = bucket.Object(key='tutorial/data.csv')
print(file)

# AWS s3 object
file.get()

# read file body (careful doing this with large files)
file.get()['Body'].read()

#Download file
s3.meta.client.download_file('public-test-bucket-d4d', 'tutorial/data.csv', 'local_data_file.csv')

# or using attributes of variables assigned earlier

s3.meta.client.download_file(bucket.name, file.key, 'local_data_new.csv')

#download all files in a s3 "folder" with specific prefix:
for item in bucket.objects.filter(Prefix='tutorial/file'):
    s3.meta.client.download_file(bucket.name, item.key, 'local_{}'.format(item.key.split('/')[-1]))

# Check dir contents after download
get_ipython().magic('ls')



