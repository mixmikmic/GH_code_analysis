import os
import requests
import json
import pprint
# python + ssl on MacOSX is rather noisy against dev clusters
requests.packages.urllib3.disable_warnings()

# set your environment variables or fill in the variables below
API_HOSTNAME = os.environ['API_HOSTNAME'] if 'API_HOSTNAME' in os.environ else '{your-cluster-hostname}'
API_USER =     os.environ['API_USER']     if 'API_USER'     in os.environ else '{api-cluster-user}'
API_PASSWORD = os.environ['API_PASSWORD'] if 'API_PASSWORD' in os.environ else '{api-cluster-password}'

# Setting up URLs and default header parameters
root_url = 'https://' + API_HOSTNAME + ':8000'

who_am_i_url   = root_url + '/v1/session/who-am-i'
login_url      = root_url + '/v1/session/login'

default_header = {'content-type': 'application/json'}

post_data = {'username': API_USER, 'password': API_PASSWORD}

resp = requests.post(login_url, 
                  data=json.dumps(post_data), 
                  headers=default_header, 
                  verify=False)

resp_data = json.loads(resp.text)

# Print the response for the login attempt.
pprint.pprint(resp_data)

default_header['Authorization'] = 'Bearer ' + resp_data['bearer_token']

# A look at the current default requests header now
pprint.pprint(default_header)

resp = requests.get(who_am_i_url, 
                  headers=default_header, 
                  verify=False)

# Print the response. Include the id, sid, and uid
pprint.pprint(json.loads(resp.text))



