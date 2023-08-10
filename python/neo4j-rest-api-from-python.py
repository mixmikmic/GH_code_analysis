import requests
import json

run_check = False
user = 'neo4j'
password = 'demo'
neo4j_server = 'http://localhost:7474'

if run_check:
    r = requests.post('{}/db/data/transaction'.format(neo4j_server), auth=(user, password))
    print(r.status_code)
    print(r.json())
    commitUrl = str(r.json()['commit'])
    commitUrl
else:
    print('Set run_check variable to True for running this code')

if run_check:
    movie_title = 'Blizzard'
    movie_tagline = 'The winds are changing'
    movie_release = 2014

    dataJson = """{
      "statements": [
        {
          "statement": "CREATE (m:Movie {title:'%s', tagline:'%s', released:%d});",
          "parameters": null,
          "resultDataContents": [
            "row",
            "graph"
          ],
          "includeStats": true
        }
      ]
    }""" % (movie_title, movie_tagline, movie_release)
    r = requests.post(commitUrl, data=dataJson, auth=(user, password))
    r.json()
else:
    print('Set run_check variable to True for running this code')

