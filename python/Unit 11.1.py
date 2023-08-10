# load the re module, we need regular expressions for tokenization
import re
# then load the Counter function from the collections module;
# we need this to count word tokens
from collections import Counter

# our example string
string1 = "The sun shone, having no alternative, on the nothing new."
print(string1)

# re.findall takes a string as input and computes a list of all the matches;
# since \w+ matches 1 or more word characters, this will split the string into a list of words;
# we also use str.lower because capitalization is misleading for word counts ("The" and "the" are the same word type)
words1 = re.findall(r"\w+", str.lower(string1))
print(words1)

# now we just feed the list words1 into Counter
counts1 = Counter(words1)
print(counts1)

import re
from collections import Counter

string1 = "The sun shone, having no alternative, on the nothing new."
string2 = "Murphy sat out of it, as though he were free, in a mew in West Brompton."
string3 = "Here for what might have been six months he had eaten, drunk, slept, and put his clothes on and off, in a medium-sized cage of north-western aspect commanding an unbroken view of medium-sized cages of south-eastern aspect."
string4 = "Soon he would have to make other arrangements, for the mew had been condemned."
string5 = "Soon he would have to buckle to and start eating, drinking, sleeping, and putting his clothes on and off, in quite alien surroundings."

# tokenize the normalized strings, count the words, and print to screen
words1 = re.findall(r"\w+", str.lower(string1))
counts1 = Counter(words1)
print(counts1)

words2 = re.findall(r"\w+", str.lower(string2))
counts2 = Counter(words2)
print(counts2)

words3 = re.findall(r"\w+", str.lower(string3))
counts3 = Counter(words3)
print(counts3)

words4 = re.findall(r"\w+", str.lower(string4))
counts4 = Counter(words4)
print(counts4)

words5 = re.findall(r"\w+", str.lower(string5))
counts5 = Counter(words5)
print(counts5)

# and now do counts for everything together
passage = string1 + " " + string2 + " " + string3 + " " + string4 + " " + string5
words = re.findall(r"\w+", str.lower(passage))
counts = Counter(words)
print(counts)

import re
from collections import Counter

string1 = "The sun shone, having no alternative, on the nothing new."
string2 = "Murphy sat out of it, as though he were free, in a mew in West Brompton."
string3 = "Here for what might have been six months he had eaten, drunk, slept, and put his clothes on and off, in a medium-sized cage of north-western aspect commanding an unbroken view of medium-sized cages of south-eastern aspect."
string4 = "Soon he would have to make other arrangements, for the mew had been condemned."
string5 = "Soon he would have to buckle to and start eating, drinking, sleeping, and putting his clothes on and off, in quite alien surroundings."


# define a custom function for counting words
def count_words(string):
    words = re.findall(r"\w+", str.lower(string))
    counts = Counter(words)
    return counts

# tokenize the normalized strings, count the words, and print to screen;
# the normalization, tokenization and word counting now all happens inside the count_words function
print(count_words(string1))
print(count_words(string2))
print(count_words(string3))
print(count_words(string4))
print(count_words(string5))

# and now do counts for everything together
passage = string1 + " " + string2 + " " + string3 + " " + string4 + " " + string5
print(count_words(passage))

# we define the function count_words with a single argument, called string
def count_words(string):
    # since string is an argument, we can use it like a variable name on the next line
    words = re.findall(r"\w+", str.lower(string))
    counts = Counter(words)
    # we have computed a value, stored as the variable counts
    # we now return this as the output of the function
    return counts

import re
from collections import Counter

string1 = "The sun shone, having no alternative, on the nothing new."

# we define the function count_words with a single argument, called string
def count_words(string):
    print("Value of string is:", string)
    words = re.findall(r"\w+", str.lower(string))
    print("Value of words is:", words)
    counts = Counter(words)
    print("Value of counts is:", counts)
    # we have computed a value, stored as the variable counts
    # we now return this as the output of the function
    return counts

print("Output of function is:", count_words(string1))

words = "This is not a list of words"

# the count_words function uses a local variable words
count_words(string1)

print("Value of words outside of function is:", words)

def double_string(string):
    return string + " " + string

print(double_string("I don't want to repeat myself!"))

def politics_filter(string):
    if "Trump" in string or "Hillary" in string:
        return "censored"
    else:
        return string
    
print(politics_filter("Vote Trump!"))
print(politics_filter("Hillary should have won!"))
print(politics_filter("Politics have no relation to morals."))

import random

def random_greeting(names):
    greeting = random.choice(["Hi, ", "Hello, "])
    name = random.choice(names)
    return greeting + name + "!"

print(random_greeting(["John", "Mary", "Sue", "Bill", "Paul", "Anne"]))

def madlibs(adjective, verb, noun):
    string = "An " + adjective + " man was " + verb + "ing his " + noun
    return string

print(madlibs("expensive", "fail", "tardiness"))

def always_do_the_same():
    return "The output of this function never changes"

print(always_do_the_same())

