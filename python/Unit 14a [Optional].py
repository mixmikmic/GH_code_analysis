import re

string = "The captain of the team is very, very angry with them!"
tokens = re.findall(r"\w+", str.lower(string))
stopwords = ["the", "a", "an", "of", "some", "in", "is", "very", "with", "at", "them", "I", "you"]

# we start defining a new list
words = []
for token in tokens:
    if token not in stopwords:
        # add token to words
        list.append(words, token)
print(words)

import re

string = "The captain of the team is very, very angry with them!"
tokens = re.findall(r"\w+", str.lower(string))
stopwords = ["the", "a", "an", "of", "some", "in", "is", "very", "with", "at", "them", "I", "you"]

# we start defining a new list
words = [token for token in tokens if token not in stopwords]
print(words)

print([num + 10 for num in [1,2,3,4,5] if num != 3])

def triple_string(string):
    return string + string + string

print([triple_string(word) for word in re.findall("\w+", "John loves Mary")])

print([2*n for n in [0,1,2,3,4,5,6,7,8,9]])

