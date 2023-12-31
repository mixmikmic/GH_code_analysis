from neo4j.v1 import GraphDatabase
import json

driver = GraphDatabase.driver("bolt://localhost:7687")

with open('./data/tweets_full.json') as json_data:
    tweetArr = json.load(json_data)

len(tweetArr)

import_query = '''
WITH $tweetArr AS tweets
UNWIND tweets AS tweet
MERGE (u:User {user_id: tweet.user_id})
ON CREATE SET u.screen_name = tweet.screen_name
MERGE (t:Tweet {tweet_id: tweet.tweet_id})
ON CREATE SET t.text = tweet.tweet_text,
              t.permalink = tweet.permalink
MERGE (u)-[:POSTED]->(t)

FOREACH (ht IN tweet.hashtags |
  MERGE (h:Hashtag {tag: ht.tag })
  ON CREATE SET h.archived_url = ht.archived_url
  MERGE (t)-[:HAS_TAG]->(h)
)

FOREACH (link IN tweet.links |
  MERGE (l:Link {url: link.url})
  ON CREATE SET l.archived_url = link.archived_url
  MERGE (t)-[:HAS_LINK]->(l)
)

'''

def add_tweets(tx):
    tx.run(import_query, tweetArr=tweetArr)

with driver.session() as session:
    session.write_transaction(add_tweets)

# GraphQL schema

graphQL_schema = '''

type Tweet {
    tweet_id: ID!
    text: String
    permalink: String
    author: User @relation(name: "POSTED", direction: "IN")
    hashtags: [Hashtag] @relation(name: "HAS_TAG", direction: "IN")
    links: [Link] @relation(name: "HAS_LINK", direction: "IN")
}

type User {
    user_id: ID!
    screen_name: String
    tweets: [Tweet] @relation(name: "POSTED", direction: "OUT")
}

type Hashtag {
    tag: ID!
    archived_url: String
    tweets: [Tweet] @relation(name: "HAS_TAG", direction: "IN")
}

type Link {
    url: ID!
    archived_url: String
    

}


'''



