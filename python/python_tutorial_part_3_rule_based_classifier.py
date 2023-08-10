import re
#-------------------
def clean_lexicon():
    positive_words= open("/Users/mam/CORE/TEACHING/smm/PROJECT-PROBLEMS/pos.swn.txt", "r").readlines()
    new_pos_list=[]
    for i in positive_words[:5]:
        i=i.strip()
        #i= i[:-1] # i is a word in the list
        i= re.sub("_", " ", i)
        new_pos_list.append(i)
    return new_pos_list

my_positive_list= clean_lexicon()
print my_positive_list[:10]

# Let's make this function more general so that we can use it to read lexical files,
# whether they are positive or negative. To do that, we simply parameterize the function.
# What this means is that we make it work with a parameter, which will be a file name that we pass to
# the function when we are calling it. Now, this parameter can be either the name of the positive lexicon file
# or the name of the negative lexicon file. So, that is a desirable change.

import re

def clean_lexicon(lex_input):
    lex_file_l=open(lex_input, "r").readlines()
    
    new_lex_file_l=[]
    for i in lex_file_l:
        i=i.strip()
        #i= i[:-1] # i is a word in the list
        i= re.sub("_", " ", i)
        new_lex_file_l.append(i)
    return new_lex_file_l

my_positive_list= clean_lexicon("/Users/mam/CORE/TEACHING/smm/PROJECT-PROBLEMS/pos.swn.txt")
print my_positive_list[:10]
print "*"*50
my_neg_list= clean_lexicon("/Users/mam/CORE/TEACHING/smm/PROJECT-PROBLEMS/neg.swn.txt")
print my_neg_list[:10]

# What if we wanted to know the percentages of positive and negative words to the overall words (tokens) in a file.
# Let's write some code to do that based on the positive and negative entries we acquired from SentiWordNet:
import re

def clean_lexicon(lex_input):
    lex_file_l=open(lex_input, "r").readlines()
    
    new_lex_file_l=[]
    for i in lex_file_l:
        i=i.strip()
        #i= i[:-1] # i is a word in the list
        i= re.sub("_", " ", i)
        new_lex_file_l.append(i)
    return new_lex_file_l

# Change the path to your local path:
pos_lex= clean_lexicon("/Users/mam/CORE/TEACHING/smm/PROJECT-PROBLEMS/pos.swn.txt")
neg_lex= clean_lexicon("/Users/mam/CORE/TEACHING/smm/PROJECT-PROBLEMS/neg.swn.txt")


# Determine the percentage of positive words in a file:
def get_sentiment_diversity(pos_lex, neg_lex, input_file):
    '''
    just returns some stats about % of pos and neg sentiment in a file...
    '''
    input_string=open(input_file, "r").read().lower()
    len_words= float(len(input_string.split()))
    pos_count=0
    neg_count=0
    for w in pos_lex:
        pos_count+= input_string.count(w)
    for w in neg_lex:
        neg_count += input_string.count(w)
    return pos_count, neg_count, len_words
   
# Call the function...
input_file="/Users/mam/CORE/TEACHING/ssa/git_hub/python_tutorial/hamlet.txt"
pos_count, neg_count, len_words= get_sentiment_diversity(pos_lex, neg_lex, input_file)
#-------------------------
print "% of positive: ", round(pos_count/len_words, 4) 
print "% of negative: ", round(neg_count/len_words, 4)

import string
punc = [char for char in string.punctuation]
print punc

# What if we wanted to remove all punctuation marks from a file?
# There are many ways to do this.
# As an introduction to regular expressions and the "string" module, let's do something along the following lines:
#----------------
# Let's take a look at the "re" module first. Here's an example:

import re
s = " hello "
print "##"+ s + "##"
s2= re.sub(" ", "", s)
print "##"+ s2 + "##"
s3=s.strip()
print "##"+ s3 + "##"


import string
import re

def clean(to_filter_list, text):
    '''
    input: 
        a. list of undesirable items we want to remove from text
        b. text we want to clean
    output:
        cleaned text
    '''
    for i in to_filter_list:
        #print i
        i="\\"+i
        text=re.sub(i, "", text)
    return text

#----------------------
# Call the function...
punc = [char for char in string.punctuation]
text="Hey there 654%$21!!!...? $& + ___ | %"

new=clean(punc, text)
print text
print new
#print punc

# Some work on filtering out undesirable content, for example retweets from a file.
# The first step is to do some analysis and understand the structure of a retweet.
# Below we assume simply thar a retweet is just a tweet that starts with either "RT" or "rt"
#--------------------------------------------
# How do we get red of retweets, for example?
# Let's say we have the following list of lines, returned from a file we opened
lines=["RT @abhi I like #soccer!!!!", "rt @abhi I cooked lentil soup",       "@alex Did you make it to the meeting?"]

new_list=[]
for line in lines:
    if not line.startswith("RT") and not line.startswith("rt"):
        #print line
        new_list.append(line)
        
print new_list

clean_list=[line for line in lines if not line.startswith("RT") and not line.startswith("rt")]
print clean_list

# Now, let's filter out duplicates:
lines=["RT @abhi I like #soccer!!!!", "rt @abhi I cooked lentil soup",       "@alex Did you make it to the meeting?",       "@alex Did you make it to the meeting...",      "@alex Did you make it there to the meeting?",      "@alex did you make it to the meeting?",      "@alex Did you maaaaake it to the meeting?"]

print set(lines)

# Why don't we now use a main function to call the code we wrote so far?


import re
import string
punc = [char for char in string.punctuation]

def clean(to_filter_list, text):
    '''
    input: 
        a. list of undesirable items we want to remove from text
        b. text we want to clean
    output:
        cleaned text
    '''
    for i in to_filter_list:
        #print i
        i="\\"+i
        text=re.sub(i, "", text)
    return text


def clean_lexicon(lex_input):
    lex_file_l=open(lex_input, "r").readlines()
    
    new_lex_file_l=[]
    for i in lex_file_l:
        i=i.strip()
        #i= i[:-1] # i is a word in the list
        i= re.sub("_", " ", i)
        new_lex_file_l.append(i)
    return new_lex_file_l


# Determine the percentage of positive words in a file:
def get_sentiment_diversity(pos_lex, neg_lex, input_file):
    '''
    just returns some stats about % of pos and neg sentiment in a file...
    '''
    input_string=open(input_file, "r").read().lower()
    input_string= clean(punc, input_string)
    len_words= float(len(input_string.split()))
    pos_count=0
    neg_count=0
    for w in pos_lex:
        pos_count+= input_string.count(w)
    for w in neg_lex:
        neg_count += input_string.count(w)
    return pos_count, neg_count, len_words
   
def main():
    # Call the code...
    #------------------
    print("Welcome to the sentiment statistician!!!")
    # Get the lexicon:
    pos_lex= clean_lexicon("/Users/mam/CORE/TEACHING/smm/PROJECT-PROBLEMS/pos.swn.txt")
    neg_lex= clean_lexicon("/Users/mam/CORE/TEACHING/smm/PROJECT-PROBLEMS/neg.swn.txt")
    # Read the hamlet file
    input_file="/Users/mam/CORE/TEACHING/ssa/git_hub/python_tutorial/hamlet.txt"
    # get sentiment stats
    pos_count, neg_count, len_words= get_sentiment_diversity(pos_lex, neg_lex, input_file)
    #-------------------------
    print "% of positive: ", round(pos_count/len_words, 4) 
    print "% of negative: ", round(neg_count/len_words, 4)

    
if __name__ == "__main__":
    main()

lines=open("/Users/mam/CORE/TEACHING/smm/PROJECT-PROBLEMS/posTweets.txt", "r").readlines()
print type(lines) 
print positive_words[0:5]
print len(positive_words)
positive_words=positive_words#+["good"]
#print lines[0:5]
pos_counter=0
for line in lines:
    for entry in positive_words:
        #print i[:-1]
        #break
        if entry in line and "never" not in line:
            #print i
            pos_counter+=1
    if pos_counter > 1:
        print("Predicted Label= POSITIVE")
    #else: #pos_counter ==0:
     #   print("No posiotive words found")
    pos_counter=0
        

lines=x[:201]
from collections import defaultdict
d=defaultdict(int)

for l in lines:
    if "good" in l and "bad" in l:
        print "mixed tweet\t", l
    elif "bad" in l:
        print "negative tweet"
    elif "good" in l:
        print "positive tweet"
    else:
        pass #print "\t\tobjective tweet"

x=open("~/Desktop/posTweets.txt", "r").readlines()
lines=x[:201]
from collections import defaultdict
d=defaultdict(int)

pos_lex=["good", "fantastic", "wonderful", "great", "fascinating", "pizza"]
neg_lex=["bad", "ugly", "boring", "disguting", "lazy"]

count_pos=0

for l in lines:
    for entry in pos_lex:
        if entry in l:
            count_pos+=1
            print count_pos #entry, lines.index(l)
    count_pos=0

