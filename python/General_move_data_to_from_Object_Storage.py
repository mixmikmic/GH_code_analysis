#!pip install --user --upgrade python-keystoneclient
#!pip install --user --upgrade python-swiftclient

credentials = {
  'auth_uri':'',
  'global_account_auth_uri':'',
  'username':'xx',
  'password':"xx",
  'auth_url':'https://identity.open.softlayer.com',
  'project':'xx',
  'project_id':'xx',
  'region':'dallas',
  'user_id':'xx',
  'domain_id':'xx',
  'domain_name':'xx',
  'tenantId':'xx'
}

import swiftclient.client as swiftclient

conn = swiftclient.Connection(
    key=credentials['password'],
    authurl=credentials['auth_url']+"/v3",
    auth_version='3',
    os_options={
        "project_id": credentials['project_id'],
        "user_id": credentials['user_id'],
        "region_name": credentials['region']})

examplefile = 'my_team_name_data_folder/zipfiles/classification_1_narrowband.zip'
etag = conn.put_object('some_container', 'classification_1_narrowband.zip', open(examplefile).read())

classification_results_file = 'my_team_name_data_folder/results/my_final_testset_classes.csv'
etag = conn.put_object('some_container', 'my_final_testset_classes.csv', open(examplefile).read())



