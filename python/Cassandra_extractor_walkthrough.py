# Add the path to the slack-pack/code/ folder in order to be able to import nlp module
import sys, os

NLP_PATH = '/'.join(os.path.abspath('.').split('/')[:-1]) + '/'
sys.path.append(NLP_PATH)

import pendulum as pm

from nlp.text import extractor as xt

reload(xt)

casdb = xt.CassandraExtractor(cluster_ips=['54.175.189.47'],
                              session_keyspace='test_keyspace',
                              table_name='awaybot_messages')

casdb.list_channels()

message_stream = casdb.get_messages(type_of_query='week', channel='data', min_words=0)

for m in message_stream:
    print m.text

message_stream = casdb.get_messages(type_of_query='week', channel='data', min_words=0)

for m in message_stream:
    print m.url

message_stream = casdb.get_messages(type_of_query='day', channel='oops', min_words=0)

for m in message_stream:
    print m.text

my_q = "select * from awaybot_messages2 where ts > '1479663861.0' and channel = 'data' ALLOW FILTERING;"
casdb.add_query(label='my_q', query=my_q)

my_stream = casdb.get_messages(type_of_query='my_q', channel='data', min_words=10)

for m in my_stream:
    print m.text



