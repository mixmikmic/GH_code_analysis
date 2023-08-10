import csv

def read_twitter_emotions(file_name):
    """ Read Twitter Emotions in Text CSV data file and return as JSON """
    print("Reading", file_name)
    data = []
    csvfile = open(file_name, "r")
    for i, line in enumerate(csv.DictReader(csvfile, delimiter=",", fieldnames=None)): # csv.DictReader returns each line as dictionary, if fieldnames are not given, it assumes that first line in the file defines fieldnames 
        if i % 1000 == 999:
            print(i+1, "tweets")
        one_example={}
        one_example["text"]=line["content"] # we have (tweet_id, sentiment, author and content) in the original data
        one_example["class"] = line["sentiment"]
        data.append(one_example)
    return data

data=read_twitter_emotions("data/text_emotion.csv")
print("Examples:", len(data))
print("Fist example:", data[0])

from collections import Counter
label_counter=Counter()
for example in data:
    label_counter.update([example["class"]]) # counter.update needs a list of new items, more efficient would be label_counter=Counter([item["class"] for item in data]), because then we update the counter only once
print("Labels:", label_counter.most_common(20))

import json

print(data[0])
with open("data/text_emotion.json","wt") as f:
    json.dump(data,f,indent=2) # indent: data will be pretty-printed with that indent level

