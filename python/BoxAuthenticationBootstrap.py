import os, sys
from boxsdk.exception import BoxAPIException

def store_tokens(access_token, refresh_token):
    
    """Callback for storing refresh tokens. (For now we ignore access tokens)."""
    with open('apptoken.cfg', 'w') as f:
        f.write(refresh_token.strip())

# Define client ID, client secret, and developer token.
CLIENT_ID = None
CLIENT_SECRET = None
REDIRECT_URI = None

# Read app info from text file
with open('app.cfg', 'r') as app_cfg:
    CLIENT_ID = app_cfg.readline()
    CLIENT_SECRET = app_cfg.readline()
    REDIRECT_URI = app_cfg.readline()

# for debugging
# print ("CLIENT_ID: ", CLIENT_ID )
# print ("CLIENT_SECRET: ", CLIENT_SECRET )
# print ("REDIRECT_URI: ", REDIRECT_URI )

from boxsdk import OAuth2

oauth = OAuth2(
    client_id=CLIENT_ID.strip(),
    client_secret=CLIENT_SECRET.strip(),
    store_tokens=store_tokens
)

auth_url, csrf_token = oauth.get_authorization_url(REDIRECT_URI.strip())

print ("auth_url", auth_url)

#access_token, refresh_token = oauth.authenticate('paste code here...')
access_token, refresh_token = oauth.authenticate('')
store_tokens(access_token, refresh_token)

from boxsdk import Client

client = Client(oauth)

root_folder = client.folder(folder_id='0').get()
print ("folder name: ", root_folder['name'] )

