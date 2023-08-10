import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from operator import itemgetter
import heapq
from copy import deepcopy

avg_reviews = []
with open("amazon-meta.txt") as amazon_file:
    for line in amazon_file:
        line = ' '.join(line.split())
        line_words = line.strip().split(' ')
        if line_words[0] == "reviews:":
            avg_reviews.append(float(line_words[-1]))

len(avg_reviews)

avg_review_counts = pd.Series(avg_reviews).value_counts()
plt.bar(avg_review_counts.index, avg_review_counts.values)
plt.show()

category_dict = {}
category_count = Counter()
with open("amazon-meta.txt") as amazon_file:
    line_cat = 0
    for line in amazon_file:
        line = ' '.join(line.split())
        line_words = line.strip().split(' ')
        if line_words[0] == "ASIN:":
            line_asin = line_words[-1]
        if (line_words[0] == "group:") & (line_words[-1] == "Book"):
            line_cat = 1
            category_dict[line_asin] = []
        elif (line_words[0] == "group:") & (line_words[-1] != "Book"):
            line_cat = 0
        if (line_cat == 1) & (line[0:6] == "|Books"):
            category_dict[line_asin].append(line.split('|')[-1])
            category_count[line.split('|')[-1]] += 1

len(category_dict)

category_dict

heapq.nsmallest(10, category_count.items(), key=itemgetter(1))

len(category_count)

least_count_categories = heapq.nsmallest(8000, category_count.items(), key=itemgetter(1))
8000/len(category_count)

smallest_cat_list = []
for i in range(0, len(least_count_categories)):
    smallest_cat_list.append(least_count_categories[i][0])
    
smallest_cat_list = set(smallest_cat_list)
smallest_cat_list

category_dict_temp = deepcopy(category_dict)
for key_j in list(category_dict_temp.keys()):
    if not(smallest_cat_list.isdisjoint(category_dict_temp[key_j])):
        del category_dict_temp[key_j]

len(category_dict_temp)/len(category_dict)

