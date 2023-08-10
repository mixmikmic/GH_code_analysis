# first, read in the data

import os
import csv

os.chdir('../data/')

records = []

with open('call_records.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        records.append(row)

print(records[0]) # print the header
records = records[1:] # remove the header
print(records[0]) # print an example record

# for each number, compute how many of its top ten match every other number

all_numbers = list(set([r[2] for r in records]))

#all_numbers = list(all_numbers)[:100] # let's work with a subset for now

# first, compute all top tens
top_tens = {}
for uniq_number in all_numbers:
    recipients = [r[3] for r in records if r[2] == uniq_number]
    uniq_recipients = sorted(list(set(recipients)))
    freq = [recipients.count(x) for x in uniq_recipients]
    top_list = [x for (x,y) in sorted(zip(uniq_recipients, freq), key=lambda pair: pair[1], reverse=True)]
    # top_list is a list of strings, with the "top" (most contacted) first (i.e descending)
    top_tens[uniq_number] = top_list

# now that we have top ten lists for every number, let's compare them

def compute_topten_similarity(number, numbers):
    our_topten = top_tens[number][:10]
    similarity_list = []
    for other_number in numbers:
        # a dilemma - do we count it as a match if it appears at all on the top ten list,
        # or just in the same positon? or maybe within +/- two of the same position?
        # for now, just compute a union
        their_topten = top_tens[other_number][:10]
        overlap = len(set(our_topten) & set(their_topten)) / 10
        similarity_list.append(1.0 - overlap)
    return similarity_list

similarity_matrix = []
for uniq_number in all_numbers:
    row = compute_topten_similarity(uniq_number, all_numbers)
    similarity_matrix.append(row)
    
get_ipython().magic('store similarity_matrix')

from matplotlib import pyplot as plt
cmap = plt.cm.Blues
heatmap = plt.pcolor(similarity_matrix, cmap=cmap)
plt.title("Overlap in Top Ten Contacts")
plt.colorbar()
fig = plt.gcf()
fig.set_size_inches(20,10)
plt.show()

from sklearn import cluster as clu
model = clu.AffinityPropagation(max_iter=1000, affinity='precomputed').fit(similarity_matrix)

cluster_centers_indices = model.cluster_centers_indices_
labels = model.labels_

n_clusters_ = len(cluster_centers_indices)

# we now have labels, and need to turn that into phone numbers
results = [[] for x in range(n_clusters_+1)]
for i,label in enumerate(labels):
    # for each label, grab the number it represents (all_numbers[i]) and
    # put it in the right cluster list (results[label])
    results[label].append(all_numbers[i])
  
for result in results:
    print(result)

