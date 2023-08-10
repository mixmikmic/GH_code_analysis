import xml.etree.ElementTree as et
import re
import os

def get_text(element):
    """ Get text within tag and within nested tags """
    return ((element.text or '') + ''.join(map(get_text, element)) + (element.tail or ''))

def read_news(filename):
    """ Extract news article texts and topic labels from Reuters XML file """
    f = open(filename, encoding="latin-1")
    output = []
    while True: # Loop through file
        buffer = ""
        while True: # Loop through article, fill buffer
            line = f.readline()
            buffer += line
            if "</REUTERS>" in line:
                break
            if not line: # End of file
                return output
        buffer = buffer.replace("&", "&amp;") # Fix XML
        root = et.fromstring(buffer) # Parse XML
        topic_tag = root.find('TOPICS')
        if len(topic_tag) != 1: # Extract only articles with exactly one topic label, for simplicity
            continue
        topic = get_text(topic_tag).strip()
        text = re.sub("\n\s+", "\n", get_text(root.find('TEXT')).strip())
        output.append({'class': topic, 'text': text}) # Save as JSON entry

data = []
path = "data/reuters"
# Read all XML (sgm) files in directory
for filename in os.listdir(path):
    if '.sgm' in filename:
        print("Reading", filename)
        data += read_news(os.path.join(path,filename))

# Check number of articles
len(data)

# Check number of classes
len(set([x['class'] for x in data]))

# Check number of articles per class
import collections
counter = collections.defaultdict(lambda: 0)
for datum in data:
    counter[datum['class']] += 1
for topic, count in sorted(counter.items(), key=lambda x:x[1], reverse=True):
    print(count, topic)
    

# Filter out classes with less than 5 occurrences
data = [d for d in data if counter[d['class']] >= 5]

# Check length
len(data)

# Check number of classes
len(set([x['class'] for x in data]))

import json
with open("data/reuters_51cls.json","w") as f:
    json.dump(data,f,indent=2)

