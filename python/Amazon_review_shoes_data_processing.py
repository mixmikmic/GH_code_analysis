import numpy as np 
import pandas as pd 

with open('Shoes.txt', 'r') as f:
    reviews = f.read()
    
# print(reviews[:1000])

# print(reviews[466] == '\n')

# print(reviews[-1500:])

review_list = reviews.split('\n\n')
# print(review_list[0])
# print()
# print(review_list[-2])
# print()
# print(review_list[-1])

# print(review_list[-1] == "")
# print(review_list[-2] == "")

review_list = review_list[:-1]

print(review_list[0])

# not all reviews end with a period or !, but they don't end with \n

test_entry = review_list[0].split('\n')
print(test_entry)
test_feature = test_entry[0].split(':')
print(test_feature)

def build_reviews_list(path):
    """ build a reviews list from the input file 
    """
    with open('Shoes.txt', 'r') as f:
        reviews = f.read()
    review_list = reviews.split('\n\n')
    if review_list[-1] == "":
        return review_list[:-1]
    else:
        return review_list

def build_review_dict(review):
    """ buuld a review dict from a review entry.
    """
    review_dict = {}
    feature_list = review.split('\n')
    for feature in feature_list:
        feature_and_content = feature.split(': ')
        review_dict[feature_and_content[0]] = feature_and_content[1]
    return review_dict

def build_list_of_review_dict(path):
    review_list = build_reviews_list(path)
    return [build_review_dict(review) for review in review_list]
    

# review_dict = build_review_dict(review_list[0])
# review_dict

reviews_list = build_list_of_review_dict('Shoes.txt')
reviews_list[0]

np.savez('shoes_list_of_review_dicts.npz', reviews_list=reviews_list)

f = np.load('shoes_list_of_review_dicts.npz')
reviews_list = f['reviews_list']
print(reviews_list[0])



