import time
from datetime import datetime
import random
import json
import os
import shutil

if 'sample' in os.listdir():
    shutil.rmtree('sample')
os.mkdir('sample')

names = ['Frodo', 'Gandalf', 'Bilbo', 'Samwise', 'Legolas', 'Aragorn']

count = 0

while True:
    # write random data to a random file
    with open('sample/stream-sample{}.txt'.format(random.random()),'w') as f:
    
        # pick a random value between 0 and 100
        random_value = random.randint(0, 100)
        
        # pick a random name from our names list
        random_name = names[random.randint(0, len(names) - 1)]
        
        # generate epoch timestamp
        timestamp = int(time.time())
        
        # generate a dictionary with some output
        data = {'count': count, 'name': random_name, 'value': random_value, 'timestamp': timestamp}
        
        # write the data to a random file
        f.write(json.dumps(data))
        
        # increment the counter
        count+=1
        
        # add one second of sleep
        time.sleep(1)

