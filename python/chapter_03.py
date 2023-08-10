import requests

base_url = 'http://chroniclingamerica.loc.gov/search/pages/results/'
parameters = '?andtext=slavery&format=json'

r = requests.get(base_url + parameters)

search_json = r.json()

print search_json.keys()

search_json['totalItems']

print type(search_json['items'])
print len(search_json['items'])

print type(search_json['items'][0])
print type(search_json['items'][19])

first_item = search_json['items'][0]

print first_item.keys()

print first_item['title']

import pandas as pd

# Make sure all columns are displayed
pd.set_option("display.max_columns",101)

df = pd.DataFrame(search_json['items'])

df.head(10)

df.to_csv('lynching_articles.csv')

base_url = 'http://chroniclingamerica.loc.gov/search/pages/results/'
parameters = '?andtext=slavery&format=json&page=3'
r = requests.get(base_url + parameters)
results =  r.json()

print results['startIndex']
print results['endIndex']

base_url = 'http://chroniclingamerica.loc.gov/search/pages/results/'
parameters = {'andtext': 'lynching',
              'page' : 3,
              'format'  : 'json'}
r = requests.get(base_url, params=parameters)

results =  r.json()

print results['startIndex']
print results['endIndex']

def get_articles():
    '''
    Make calls to the Chronicling America API.
    '''
    base_url = 'http://chroniclingamerica.loc.gov/search/pages/results/'
    parameters = {'andtext': 'lynching',
                  'page' : 3,
              'format'  : 'json'}
    r = requests.get(base_url, params=parameters)
    results =  r.json()
    return results

results = get_articles()

print results['startIndex']
print results['endIndex']

def get_articles(search_term, page_number):
    '''
    Make calls to the Chronicling America API.
    '''
    base_url = 'http://chroniclingamerica.loc.gov/search/pages/results/'
    parameters = {'andtext': search_term,
                  'page' : page_number,
              'format'  : 'json'}
    r = requests.get(base_url, params=parameters)
    results =  r.json()
    return results

results = get_articles('lynching', 3)

print results['startIndex']
print results['endIndex']

for page_number in range(1,4): # range stops before it gets to the last number
    results = get_articles('lynching', page_number)
    print results['startIndex'], results['endIndex']
    

df = pd.DataFrame()

for page_number in range(1,4):
    results = get_articles('lynching', page_number)
    new_df = pd.DataFrame(results['items'])
    df = df.append(new_df , ignore_index=True)
    
print len(df)
df.head(5)

import oauth2

consumer_key    = 'qDBPo9c_szHVrZwxzo-zDw'
consumer_secret = '4we8Jz9rq5J3j15Z5yCUqmgDJjM'
token           = 'jeRrhRey_k-emvC_VFLGrlVHrkR4P3UF'
token_secret    = 'n-7xHNCxxedmAMYZPQtnh1hd7lI'

consumer = oauth2.Consumer(consumer_key, consumer_secret)

category_filter = 'restaurants'
location = 'Chapel Hill, NC'
options =  'category_filter=%s&location=%s&sort=1' % (category_filter, location)
url = 'http://api.yelp.com/v2/search?' + options

oauth_request = oauth2.Request('GET', url, {})
oauth_request.update({'oauth_nonce'      : oauth2.generate_nonce(),
                      'oauth_timestamp'  : oauth2.generate_timestamp(),
                      'oauth_token'       : token,
                      'oauth_consumer_key': consumer_key})

token = oauth2.Token(token, token_secret)
oauth_request.sign_request(oauth2.SignatureMethod_HMAC_SHA1(), consumer, token)
signed_url = oauth_request.to_url()

print signed_url

resp = requests.get(url=signed_url)
chapel_hill_restaurants = resp.json()

print chapel_hill_restaurants.keys()

chapel_hill_restaurants['businesses'][1]

print chapel_hill_restaurants['total']
print len(chapel_hill_restaurants['businesses'])

def get_yelp_page(location, offset):
    '''
    Retrieve one page of results from the Yelp API
    Returns a JSON file
    '''
    # from https://github.com/Yelp/yelp-api/tree/master/v2/python
    consumer_key    = 'qDBPo9c_szHVrZwxzo-zDw'
    consumer_secret = '4we8Jz9rq5J3j15Z5yCUqmgDJjM'
    token           = 'jeRrhRey_k-emvC_VFLGrlVHrkR4P3UF'
    token_secret    = 'n-7xHNCxxedmAMYZPQtnh1hd7lI'
    
    consumer = oauth2.Consumer(consumer_key, consumer_secret)
    
    url = 'http://api.yelp.com/v2/search?category_filter=restaurants&location=%s&sort=1&offset=%s' % (location, offset)
    
    oauth_request = oauth2.Request('GET', url, {})
    oauth_request.update({'oauth_nonce': oauth2.generate_nonce(),
                          'oauth_timestamp': oauth2.generate_timestamp(),
                          'oauth_token': token,
                          'oauth_consumer_key': consumer_key})
    
    token = oauth2.Token(token, token_secret)
    
    oauth_request.sign_request(oauth2.SignatureMethod_HMAC_SHA1(), consumer, token)
    
    signed_url = oauth_request.to_url()
    resp = requests.get(url=signed_url)
    return resp.json()

def get_yelp_results(location):
    '''
    Retrive both pages of results from the Yelp API
    Returns a dataframe
    '''
    df = pd.DataFrame()
    for offset in [1,21]:
        results = get_yelp_page(location, offset)
        new_df = pd.DataFrame(results['businesses'])
        df = df.append(new_df , ignore_index=True)
    return df

ch_df = get_yelp_results('Chapel Hill, NC')

print len(ch_df)

ch_df.keys()

ch_df[['name','categories','review_count','rating']].sort_values(by='rating', ascending=False)

chapel_hill_restaurants = get_yelp_businesses('Chapel Hill, NC', 21)
for business in chapel_hill_restaurants['businesses']:
    print '%s - %s (%s)' % (business['rating'], business['name'], business['review_count'])

beverly_hills_restaurants = get_yelp_businesses('90210')
for business in beverly_hills_restaurants['businesses']:
    print '%s - %s (%s)' % (business['rating'], business['name'], business['review_count'])

pwd



