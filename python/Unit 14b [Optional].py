# a list can contain multiple elements
example_list = ["the", "boy", "likes", "the", "girl"]
# converting the list to a set removes all duplicates
example_set = set(example_list)

print(example_list)
print(example_set)

# a list has a specific order
example_list = ["first", "second", "third", "fourth"]
# converting the list to a set destroys the order
example_set = set(example_list)

print("Printing list items")
for item in example_list:
    print(item)
    
print("\nPrinting set items")
for item in example_set:
    print(item)

def char_equivalent(string1, string2):
    # convert strings to sets of characters
    string1 = set(string1)
    string2 = set(string2)
    if string1 == string2:
        return True
    else:
        return False
    
# let's run some tests:

# the comparison is case sensitive
print(char_equivalent("Tokyo", "Kyoto"))

# but order of characters does not matter, as desired
print(char_equivalent("tokyo", "kyoto"))

# and repetition is also fine
print(char_equivalent("New York", "New New York"))

# run this cell first to define the necessary variables

import urllib.request
import re
from collections import Counter

# we first define custom functions for all individual steps

def get_file(text):
    if text == "hamlet":
        urllib.request.urlretrieve("http://www.gutenberg.org/cache/epub/1524/pg1524.html", "hamlet.txt")
    if text == "faustus":
        urllib.request.urlretrieve("http://www.gutenberg.org/cache/epub/811/pg811.txt", "faustus.txt")
    if text == "johncarter":
        urllib.request.urlretrieve("http://www.gutenberg.org/cache/epub/62/pg62.txt", "johncarter.txt")
        
def read_file(filename):
    with open(filename, "r", encoding="utf-8") as text:
        return text.read()
    
def delete_before_line(string, line):
    return str.split(string, "\n", line)[-1]

def delete_after_line(string, line):
    return str.join("\n", str.split(string, "\n")[:line+1])

def hamlet_cleaner(text):
    # 0. delete unwanted lines
    text = delete_after_line(delete_before_line(text, 366), 10928)
    # 1. remove all headers, i.e. lines starting with <h1, <h2, <h3, and so on
    text = re.sub(r"<h[0-9].*", r"", text)
    # 2. remove speaker information, i.e. lines of the form <p id="id012345789"...<br/>
    text = re.sub(r'<p id="id[0-9]*">[^<]*<br/>', r"", text)
    # 3. remove html tags, i.e. anything of the form <...>
    text = re.sub(r"<[^>]*>", r"", text)
    # 4. remove anything after [ or before ] on a line (this takes care of stage descriptions)
    text = re.sub(r"\[[^\]\n]*", r"", text)
    text = re.sub(r"[^\[\n]*\]", r"", text)
    return text

def faustus_cleaner(text):
    # 0. delete unwanted lines
    text = delete_after_line(delete_before_line(text, 139), 2854)
    # 1. remove stage information
    #    (anything after 10 spaces)
    text = re.sub(r"(\s){10}[^\n]*", r"", text)
    # 2. remove speaker information
    #    (any word in upper caps followed by space or dot)
    text = re.sub(r"[A-Z]{2,}[\s\.]", r"", text)
    # 3. remove anything between square brackets (this takes care of footnote markers)
    text = re.sub(r"\[[^\]]*\]", r"", text)
    return text

def johncarter_cleaner(text):
    # 0. delete unwanted lines
    text = delete_after_line(delete_before_line(text, 234), 6940)
    # 1. delete CHAPTER I
    # (must be done like this because Roman 1 looks like English I)
    text = re.sub("CHAPTER I", "", text)
    # 2. remove any word in upper caps that is longer than 1 character
    text = re.sub(r"[A-Z]{2,}", r"", text)
    # 3. remove anything after [ or before ] on a line
    text = re.sub(r"\[[^\]\n]*", r"", text)
    text = re.sub(r"[^\[\n]*\]", r"", text)
    return text

def tokenize(string):
    return re.findall(r"\w+", string)

def count(token_list):
    return Counter(token_list)


# and now we have two functions that use all the previous functions
# to do all the necessary work for us
def get_and_clean(text):
    get_file(text)
    string = read_file(text + ".txt")
    string = str.lower(string)
    # file-specific cleaning steps
    if text == "hamlet":
        return hamlet_cleaner(string)
    if text == "faustus":
        return faustus_cleaner(string)
    if text == "johncarter":
        return johncarter_cleaner(string)

hamlet_full = tokenize(get_and_clean("hamlet"))

# define stop words
urllib.request.urlretrieve("https://raw.githubusercontent.com/Alir3z4/stop-words/master/english.txt", "stopwords.txt")
stopwords_list = re.findall(r"[^\n]+", read_file("stopwords.txt"))
stopwords_set = set(stopwords_list)

def test_list():
    # empty list of words
    words = []

    # start for-loop
    for token in hamlet_full:
        if token not in stopwords_list:
            # add token to words
            list.append(words, token)
        
# tell Jupyter to time how long it takes to run the function
get_ipython().run_line_magic('time', 'test_list()')

def test_set():
    # empty list of words
    words = []

    # start for-loop
    for token in hamlet_full:
        if token not in stopwords_set:
            # add token to words
            list.append(words, token)
        
# tell Jupyter to time how long it takes to run the function
get_ipython().run_line_magic('time', 'test_set()')

url = "https://raw.githubusercontent.com/dwyl/english-words/master/words.txt"
urllib.request.urlretrieve(url, "words.txt")
dict_string = read_file("words.txt")

dict_list = re.findall("[^\n]+", dict_string)
dict_set = set(dict_list)

def test_list():
    # empty list of words
    words = []

    # start for-loop
    for token in hamlet_full[:10000]:
        if token not in dict_list:
        # add token to words
            list.append(words, token)
        
# tell Jupyter to time how long it takes to run the function
get_ipython().run_line_magic('time', 'test_list()')

def test_set():
    # empty list of words
    words = []

    # start for-loop
    for token in hamlet_full:
        if token not in dict_set:
        # add token to words
            list.append(words, token)
        
# tell Jupyter to time how long it takes to run the function
get_ipython().run_line_magic('time', 'test_set()')

