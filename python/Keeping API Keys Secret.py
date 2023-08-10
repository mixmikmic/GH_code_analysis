# See above - we've created an API key in a file called 'apikey', 
# in the same directory as this notebook
filename = 'apikey'

def get_file_contents(filename):
    """ Given a filename,
        return the contents of that file
    """
    try:
        with open(filename, 'r') as f:
            # It's assumed our file contains a single line,
            # with our API key
            return f.read().strip()
    except FileNotFoundError:
        print("'%s' file not found" % filename)

api_key = get_file_contents(filename)
print("Our API key is: %s" % (api_key))

get_ipython().system('pip install pyyaml')

import yaml
TWITTER_CONFIG_FILE = 'auth.yaml'

with open(TWITTER_CONFIG_FILE, 'r') as config_file:
    config = yaml.load(config_file)
    
print(type(config))

import json

print(json.dumps(config, indent=4, sort_keys=True))

consumer_key = config['twitter']['consumer_key']
print(consumer_key)

