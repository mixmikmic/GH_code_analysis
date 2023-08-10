# install neo4j-driver
get_ipython().system('pip install neo4j-driver')

from neo4j.v1 import GraphDatabase, basic_auth
import requests
import json

OSCON_FEED_URL = "http://conferences.oreilly.com/oscon/open-source-us/public/content/report/schedule_feed"

r = requests.get(OSCON_FEED_URL)
d = r.json()

d['Schedule'].keys()

INSERT_EVENTS_QUERY = '''
WITH {events} AS events
UNWIND events AS event
MERGE (t:Talk {serial: event.serial})
  ON CREATE SET t.name = event.name,
    t.type = event.event_type,
    t.time_start = event.time_start,
    t.time_stop = event.time_stop,
    t.description = event.description,
    t.url = event.website_url,
    t.image = event.large_img_url,
    t.youtube_url = event.youtube_url

MERGE (r:Room {serial: event.venue_serial})
CREATE UNIQUE (t)-[:IN]->(r)

FOREACH (speaker IN event.speakers |
  MERGE (s:Speaker {serial: speaker})
  CREATE UNIQUE (s)-[:PRESENTS]->(t)
)

FOREACH (cat in event.categories |
  MERGE (top:Topic {name: cat})
  CREATE UNIQUE (t)-[:ABOUT]->(top)
  
  MERGE (trac:Track {name: cat})
  CREATE UNIQUE (t)-[:PART_OF]->(trac)
) 
'''

INSERT_SPEAKERS_QUERY = '''
WITH {speakers} AS speakers
UNWIND speakers AS speaker
MERGE (s:Speaker {serial: speaker.serial})
  SET s.name = speaker.name,
    s.photo = speaker.photo,
    s.url = speaker.url,
    s.position = speaker.position,
    s.twitter = speaker.twitter,
    s.bio = speaker.bio,
    s.image = speaker.large_img_url,
    s.youtube = speaker.youtube_url

WITH s,speaker WHERE speaker.affiliation IS NOT NULL
MERGE (org:Organization {name: speaker.affiliation})
CREATE UNIQUE (s)-[:AFFILIATED]->(org)
'''

INSERT_VENUES = '''
WITH {venues} AS venues
UNWIND venues AS venue
MERGE (r:Room {serial: venue.serial})
SET r.name = venue.name

'''

# instantiate the driver and create a session
driver = GraphDatabase.driver("bolt://localhost", auth=basic_auth("neo4j", "letmein"))
session = driver.session()

session.run(INSERT_EVENTS_QUERY, parameters={'events': d['Schedule']['events']})

session.run(INSERT_SPEAKERS_QUERY, parameters={'speakers': d['Schedule']['speakers']})

session.run(INSERT_VENUES,parameters={'venues': d['Schedule']['venues']})

# What organizations in "The New Stuff" track have the most speakers?
result = session.run('''
    MATCH (track:Track {name: "The New Stuff"}) 
    MATCH (track)<-[:PART_OF]-(t:Talk)<-[:PRESENTS]-(s:Speaker)
    MATCH (s)-[:AFFILIATED]->(org:Organization)
    RETURN org.name AS organization, count(*) AS num 
    ORDER BY num DESC LIMIT 10
''')

for record in result:
    print("%s: %s" % (record['organization'],record['num']))



