url = "https://pub.orcid.org/0000-0002-2907-3313/orcid-works"

import requests
import json

resp = requests.get(url,
                    headers={'Accept':'application/orcid+json'})

orcid_data = resp.json()

works =  orcid_data['orcid-profile']['orcid-activities']['orcid-works']['orcid-work']

#for key in orcid_data.keys():
#    print(key)

works

def dic_item(item):
    dobj={}
    if item['work-external-identifiers'] and item['work-citation']:
        doi = item['work-external-identifiers']['work-external-identifier'][0]['work-external-identifier-id']['value']
        dobj['cite'] = item['work-citation']['citation']
        dobj['url'] = item['url']
        dobj['journal'] = item['journal-title']
        return doi, dobj,
    else:
        return None, None

d = {}
for n,item in enumerate(works):
    doi, tmp_d =  dic_item(item)
    if doi:
        print(n, doi)
        d[doi] = tmp_d
        

for item in d.values():
    if item:
        print(item['cite'],'\n')
    else:
        pass

#from pybtex.database.input import bibtex

d['0036-9241']

import logging
logging.basicConfig(level=logging.DEBUG)

def _get_raw_json(url):
    """Get raw JSON file for orcid_id."""
    #url = orcid_url(orcid_id)
    url = "https://pub.orcid.org/0000-0002-2907-3313"
    logging.info(url)
    resp = requests.get(url,
                        headers={'Accept':'application/orcid+json'})

    return resp.json()

auth_raw = _get_raw_json(url)

auth_raw


def get_json(orcid_id):
    """Get JSON for Orcid and clean it."""
    raw_json = _get_raw_json(orcid_id)

    # TODO Add information
    myjson = {
            "given_names":
            raw_json.get("orcid-profile").get("orcid-bio").get("personal-details").get("given-names").get("value"),
            "family_name":
            raw_json.get("orcid-profile").get("orcid-bio").get("personal-details").get("family-name").get("value"),
            "affiliation": None,
            "summary": None,
            "doiurl": None,
            "title": None,
            "gravatarhash": None,
            }

    return myjson

bio = auth_raw.get('orcid-profile').get('orcid-bio').get('biography').get('value')

auth_raw.get('email')

auth_raw['orcid-profile']['orcid-bio']

auth.get("orcid-profile").get("orcid-bio").get("personal-details").get("given-names")

email = auth_raw.get("orcid-profile").get("orcid-bio").get("contact-details").get("email")[0].get("value").lower().strip()

first_name = auth_raw.get("orcid-profile").get("orcid-bio").get("personal-details").get("given-names").get("value")

" ".join(["one","two"])

test = " aa@aa.com "

test.strip()

email

import hashlib

import hashlib
gravatarhash = hashlib.md5(email.encode('utf-8')).hexdigest()



#import re
orcid = '999-999-999-999'
if len(orcid) is not 23:
    None
    return render_template('sample.html',
                           feedback={
                           "title": "Sorry page can't be created:",
                           "details": 'ORCID provided is not correct length'})





