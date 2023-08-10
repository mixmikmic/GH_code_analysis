from mapr_streams_python import Producer
p = Producer({'streams.producer.default.stream': '/user/mapr/mystream:mytopic'})
some_data_source= ["msg1", "msg2", "msg4", '\0']
for data in some_data_source:
    p.produce('/user/mapr/mystream:mytopic', data.encode('utf-8'))
p.flush()

# Consumer
from mapr_streams_python import Consumer, KafkaError
c = Consumer({'group.id': 'mygroup',
              'default.topic.config': {'auto.offset.reset': 'earliest'}})
c.subscribe(['/user/mapr/mystream:mytopic'])
running = True
while running:
  msg = c.poll(timeout=1.0)
  if msg is None: continue
  if not msg.error():
    msg_value = msg.value().decode('utf-8')
    if msg_value  is '\0':
      running = False
    print('Received message: %s' % msg_value)
  elif msg.error().code() != KafkaError._PARTITION_EOF:
    print(msg.error())
    running = False
c.close()





