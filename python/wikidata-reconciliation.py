import pandas as pd

gaz = pd.read_csv('osmanli-yer-adlari.tsv',sep='\t')

gaz.head()

gaz[gaz.Wikidata_url.notnull()].head()

import re

re.sub(r'wiki\/(.+)$',r'wiki/Special:EntityData/\1.json','https://www.wikidata.org/wiki/Q42')

gaz['wikidata_json'] = gaz.Wikidata_url.replace(r'wiki\/(.+)$',r'wiki/Special:EntityData/\1.json',regex=True)

test = gaz[gaz.wikidata_json.notnull()].wikidata_json[6]

import requests

response = requests.get(test)

response.json()

def get_wikidata_coordinates(url):
    """
    Takes a Wikidata json url and returns a pair of coordinates if they exist
    or a pair of Nones if no coordinates were found.
    """
    if pd.isnull(url):
        return None,None
    try:
        response = requests.get(url)
        print('Parsing {}'.format(url))
        data = response.json()
        for k,entity in data['entities'].items():
            for l,claim in entity['claims'].items():
                try:
                    for c in claim:
                        lat = c['mainsnak']['datavalue']['value']['latitude']
                        lon = c['mainsnak']['datavalue']['value']['longitude']
                    return lat, lon
                except:
                    continue
            return None, None
    except requests.exceptions.MissingSchema:
        print('{} is not a URL.'.format(url))
        return None,None
    except:
        return None,None

get_wikidata_coordinates(test)

gaz['lat'],gaz['lon'] = zip(*gaz.wikidata_json.map(get_wikidata_coordinates))

gaz.to_csv('osmanli-yer-adlari-with-coords.tsv',sep='\t',)

