def store_tokens(access_token, refresh_token):
    
    """Callback for storing refresh tokens. (For now we ignore access tokens)."""
    with open('apptoken.cfg', 'w') as f:
     f.write(refresh_token.strip())

import os

CLIENT_ID = None
CLIENT_SECRET = None
REDIRECT_URI = None
os.chdir('/global/home/users/mmanning')
# Read app info from text file
with open('app.cfg', 'r') as app_cfg:
    CLIENT_ID = app_cfg.readline()
    CLIENT_SECRET = app_cfg.readline()
    REDIRECT_URI = app_cfg.readline()

REFRESH_TOKEN = None

# Read app info from text file
with open('apptoken.cfg', 'r') as apptoken_cfg:
    REFRESH_TOKEN = apptoken_cfg.readline()

from boxsdk import OAuth2
from boxsdk import Client

# Do OAuth2 authorization.
oauth = OAuth2(
    client_id=CLIENT_ID.strip(),
    client_secret=CLIENT_SECRET.strip(),
    refresh_token=REFRESH_TOKEN.strip(),
    store_tokens=store_tokens
)

client = Client(oauth)

root_folder = client.folder(folder_id='0').get()
print ("folder name: ", root_folder['name'] )

items = client.folder(folder_id='0').get_items(limit=100, offset=0)
#print ("items: ", items )


import os
import shutil 

print ('current working directory: ', os.getcwd())
os.chdir('/global/scratch/user_name_here/test')

# test folder contents
items = client.folder(folder_id='0').get_items(limit=20, offset=0)
if type(items) is list:
    print ('number of files in top folder: ', len(items) )
    
    for item in items:
        if item['type'] == 'folder':
            print('folder name: ', item['name'])
        # download all image files
        if  not item['type'] == 'folder' and item['name'].endswith('.jpg'):
            imagecontent = client.file(file_id=item['id']).content()
            newfile = open('/global/scratch/user_name_here/' + item['name'], 'wb')
            newfile.write(imagecontent)
            newfile.close()

newFolder = client.folder(folder_id='0').create_subfolder('ThisIsATest')
newFolderId = newFolder['id']
sourceFolder = '/global/scratch/user_name_here/'

print ("new folder id: ", newFolderId )

upload_folder = client.folder(folder_id=newFolderId).get()

# upload all the files in the current folder  if os.path.isfile(f)
files = [f for f in os.listdir(sourceFolder) ] 
print ("files: ", files )

for filename in files:
    print ("file name: ", filename ) 
    
    if filename.endswith('.jpg'):
        upload_folder.upload(sourceFolder + filename)  

