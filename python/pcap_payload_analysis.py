import pandas as pd
import numpy as np

from io import StringIO
import re

# !tcpdump -nttxv -r test.pcap > test.txt

pcapfile='data/test.txt'

lines_tobe_printed = 10

with open(pcapfile) as myfile:
    firstlines=myfile.readlines()[0:lines_tobe_printed] #put here the interval you want
    for x in firstlines:
        print(x.strip())

sio = StringIO()
fast_forward = True
payload=''
appended_data = []

with open(pcapfile, 'rb') as f:
    for line in f:
        line = line.decode("utf-8").strip() # converting Bytes in utf-8

        if re.match(r'0x\d+', line): #Getting lines that have payload
            if line.startswith('0x0000'):
                appended_data.append(payload.strip())
                #print(payload.strip()) #DEBUG: Print the concatenated previous payload (without blank space in the beginning)
                payload = ''
            payload = payload + ' '+ line.split(':  ')[1] 
    appended_data.pop(0) # Removing the first line that is an empty line 

type(appended_data)

appended_data[0:4]

df = pd.DataFrame(appended_data)

df.head()



