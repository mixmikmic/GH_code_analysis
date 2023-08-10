# This is how you query the Slack team for all channels 
# TODO: See if DM channels are listed using different api call

import os
import time
from slackclient import SlackClient

token = 'your-token-here' 


channels = [channel_dict['id'] for channel_dict in sc.api_call("channels.list")['channels']]
print channels
    

get_ipython().run_cell_magic('writefile', 'example.py', '#!/home/kevin/slackpstone/bin/python\nimport threading, logging, time\n\nfrom kafka import KafkaConsumer, KafkaProducer\n\n# Replace the #! shebang with your env\n# This is a simple Kafka setup using python\n# On one thread we set up a producer and a topic called \'my-topic\' and send two messages each second\n# Example from https://github.com/dpkp/kafka-python/blob/master/example.py\n\n# On another thread we set up a consumer and read the topic\nclass Producer(threading.Thread):\n    daemon = True\n\n    def run(self):\n        producer = KafkaProducer(bootstrap_servers=\'localhost:9092\')\n\n        while True:\n            producer.send(\'my-topic\', b"test")\n            producer.send(\'my-topic\', b"\\xc2Hola, mundo!")\n            time.sleep(1)\n\n\nclass Consumer(threading.Thread):\n    daemon = True\n\n    def run(self):\n        consumer = KafkaConsumer(bootstrap_servers=\'localhost:9092\',\n                                 auto_offset_reset=\'earliest\')\n        consumer.subscribe([\'my-topic\'])\n\n        for message in consumer:\n            print (message)\n\n\ndef main():\n    threads = [\n        Producer(),\n        Consumer()\n    ]\n\n    for t in threads:\n        t.start()\n\n    time.sleep(20)\n\nif __name__ == "__main__":\n    logging.basicConfig(\n        format=\'%(asctime)s.%(msecs)s:%(name)s:%(thread)d:%(levelname)s:%(process)d:%(message)s\',\n        level=logging.INFO\n        )\nmain()')

get_ipython().system('python example.py')

# This is how you read from channel history
# In this case we write to file

output_example =  open('slackpstone-channel-output.txt', 'w')

for channel in channels:
    channel_history = sc.api_call("channels.history", channel=channel, count="100000")
    for message_dict in channel_history['messages']:
        if 'user' in message_dict:
            output_example.write('{}\t{}\t{}\n'.format(
                message_dict['text'].replace('\n','').encode('utf-8'),
                    message_dict['user'], message_dict['ts']))
output_example.close()

# Example of the stuff we wrote to file
c = 0
with open('slackpstone-channel-output.txt', 'r') as f:
    for line in f:
        if c<10:
            print line.strip().split('\t')
            c += 1
        else:
            break

# This is how you get the team id from the slack api
sc.api_call('team.info')['team']['id']

get_ipython().run_cell_magic('writefile', 'slack_example.py', '#!/home/kevin/slackpstone/bin/python\n\n# Integrating slack api and kafka\nfrom slackclient import SlackClient\nfrom kafka import KafkaConsumer, KafkaProducer\nimport threading, logging, time\n\nproducer = KafkaProducer(bootstrap_servers=\'localhost:9092\')\nc = 0\ntoken = \'your-token-here\'\nsc = SlackClient(token)\nteam_id = sc.api_call(\'team.info\')[\'team\'][\'id\']\n\n# First we go through all the history\n# I\'m using the team_id as the topic name\nchannels = [channel_dict[\'id\'] for channel_dict in sc.api_call("channels.list")[\'channels\']]\nfor channel in channels:\n    channel_history = sc.api_call("channels.history", channel=channel, count="100000")\n    for message_dict in channel_history[\'messages\']:\n        if \'user\' in message_dict:\n            message = \'{}\\t{}\\t{}\\t{}\\n\'.format(\n                message_dict[\'text\'].replace(\'\\n\',\'\').encode(\'utf-8\'),\n                channel, message_dict[\'user\'], message_dict[\'ts\'])\n            producer.send(team_id, message)\n            c += 1\n\n# Second, we set up a Real Time Messaging API connection and listen for text messages\n# TODO: Look into serialization with avro\n# TODO: Look at encoding issues\n# TODO: Iterate on message structure, what if any other messages we would like to send to kafka\n# TODO: Look at emoji, reactions, etc:\nif sc.rtm_connect():\n    while True:\n        latest = sc.rtm_read()\n        if latest:\n            if \'text\' in latest[0]:\n                message = \'{}\\t{}\\t{}\\t{}\\n\'.format(\n                    latest[0][\'text\'].replace(\'\\n\',\'\').encode(\'utf-8\'), \n                    latest[0][\'channel\'], latest[0][\'user\'],\n                    latest[0][\'ts\'])\n                producer.send(team_id, message)\n                c += 1\n                print \'Sent {} messages\'.format(c)\n        time.sleep(5)')

get_ipython().system('python slack_example.py')

