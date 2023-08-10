import os
import time
from slackclient import SlackClient

# add your token and start up the SlackClient
# token can be obtained here: https://api.slack.com/docs/oauth-test-tokens
token = 'TOKEN!!!'
sc = SlackClient(token)

# print out all the channels in the given team for the Slack token
channels = []
for channel_dictionary in sc.api_call("channels.list")['channels']:
    channels.append(channel_dictionary)

# for each channel, print the channel id and the channel name
for channel in channels:
    print "Channel name:", channel['name']
    print "Channel id:", channel['id']
    print "\n"

get_ipython().run_cell_magic('writefile', 'pullSlack.py', "#!/usr/bin/python\n\n# import the Kafka consumers and producers \n# like Kevan did in his example\nimport threading, logging, time\nfrom kafka import KafkaConsumer, KafkaProducer\n\n# create a Kafka producer, something that writes to Kafka\nclass Producer(threading.Thread):\n    daemon = True\n\n    # the run function that will actually be the method that \n    # writes to Kafka\n    def run(self, topic, writing):\n        producer = KafkaProducer(bootstrap_servers='localhost:9092')\n        \n        # send the topic and writing that was sent\n        producer.send(topic, writing)\n\n\ndef main():\n            \n    # start the producer\n    Producer.start()\n\n    # check every second for new messages\n    if sc.rtm_connect():\n        while True:\n            new = sc.rtm_read()\n            time.sleep(1)\n\n            # if we actually got a message\n            if len(new) > 0:\n\n                # go through each message and \n                # grab the channel and the message\n                for message in new:\n                    \n                    # try to grab the message and the\n                    # channel and also try to write\n                    # it to Kafka\n                    try: \n                        text = message['text']\n                        channel = message['channel']\n                        Producer.run(channel,text)\n                    \n                    # we'll pass if that doesn't work\n                    # because the information we got was\n                    # likely not a message but just a \n                    # status update, e.g. someone going\n                    # from away to active status\n                    except:\n                        pass\nmain()")



