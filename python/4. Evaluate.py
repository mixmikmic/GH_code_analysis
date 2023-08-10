# first, read in the data

import os
import csv

os.chdir('../data/')

records = []

with open('solutions.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        records.append(row)

print(records[0]) # print the header
records = records[1:] # remove the header
print(records[0]) # print an example record

def generate_labels(cluster):
    """ given a list of phone numbers (as strings), return a list of category labels (integers)
    that correspond to those numbers """
    all_people = list(set([r[0] for r in records]))
    categories = range(len(all_people))
    labels = []
    for number in cluster:
        person = [r[0] for r in records if r[1] == number]
        if len(person) != 1:
            raise ValueError("shouldn't be more or less than one person per number")
        person = person[0]
        labels.append(all_people.index(person))
    return labels

from sklearn import metrics

# retrieve our clustered data from "3. Training"
get_ipython().magic('store -r all_numbers')
get_ipython().magic('store -r labels')

labels_true = generate_labels(all_numbers)

metrics.adjusted_rand_score(labels_true, labels)

