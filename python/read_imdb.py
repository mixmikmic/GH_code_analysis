import csv

def read_imdb(file_name):
    """ Read IMDB Sentiment CSV data file and return as JSON """
    print("Reading", file_name)
    data = []
    csvfile = open(file_name, 'r')
    for i, line in enumerate(csv.DictReader(csvfile, delimiter="\t")):
        if i % 1000 == 999:
            print(i+1, "comments")
        one_example={}
        one_example["text"]=line['review']
        if 'sentiment' in line:
            one_example['class'] = line['sentiment']
        data.append(one_example)
    return data

data_train = read_imdb("data/imdb_train.tsv")
print(data_train[0])

import csv

def read_imdb(file_name):
    """ Read IMDB Sentiment CSV data file and return as JSON """
    print("Reading", file_name)
    data = []
    csvfile = open(file_name, 'r')
    for i, line in enumerate(csv.DictReader(csvfile, delimiter="\t")):
        if i % 1000 == 999:
            print(i+1, "comments")
        one_example={}
        one_example["text"]=line['review'].replace("<br />"," ") # Replacement happens here
        if 'sentiment' in line:
            if line['sentiment']=='1':
                one_example['class'] = 'pos'
            elif line['sentiment']=='0':
                one_example['class'] = 'neg'
            else:
                assert False, ("Unknown sentiment", line['sentiment'])
        data.append(one_example)
    return data

data_train = read_imdb("data/imdb_train.tsv")
print(data_train[0])

import json

print(data_train[0])
with open("data/imdb_train.json","wt") as f:
    json.dump(data_train,f,indent=2)

