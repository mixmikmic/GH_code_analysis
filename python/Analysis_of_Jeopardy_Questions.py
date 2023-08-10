import pandas as pd

# Dataset compiled and shared on reddit: https://www.reddit.com/r/datasets/comments/1uyd0t/200000_jeopardy_questions_in_a_json_file/
# Here we will use the full length file.
jeopardy = pd.read_csv("C:\\Users\\User\\Documents\\Python projects\\JEOPARDY_CSV.csv")

print(jeopardy.head())
print('')
print(jeopardy.columns.tolist())

#Remove spaces from column names
cols = jeopardy.columns.tolist()
for i,c in enumerate(cols):
    cols[i] = str.replace(c," ","")
jeopardy.columns = cols

print("New column names:")
print(jeopardy.columns.tolist())

import string

def norm_string(instr):
    #To lowercase
    instr = instr.lower()
        
    #Using string translate function to remove all punctuation
    translator = str.maketrans('', '', string.punctuation)
    instr = instr.translate(translator)
       
    return instr

jeopardy['clean_question'] = jeopardy['Question'].apply(norm_string)
jeopardy['clean_answer'] = jeopardy['Answer'].apply(norm_string)
jeopardy.head()

# Let's have a quick look for the longest words in the question column before moving on. 
# The longest dictionary word is 45 letters long, so if we're seeing words longer than that we like have dirty data.
def longest_word(instr):
    str_list = instr.split(' ')
    len_list = [len(x) for x in str_list]
    return max(len_list)

jeopardy['longest_quest_word'] = jeopardy['clean_question'].apply(longest_word)

import matplotlib.pyplot
get_ipython().run_line_magic('matplotlib', 'inline')

jeopardy.hist('longest_quest_word')

# A bit of a cluster above 30. Likely errors. Let's have a look at a few:
# Setting > 31 as boundary after exploring the values individually and finding a valid word at 31 characters
jeopardy[jeopardy['longest_quest_word'] > 31].head()

# Definitely dirty looking questions. We could possibly attempt to scrape the listed websites for more info, 
# but for the moment let's drop these rows and continue.
jeopardy = jeopardy[jeopardy['longest_quest_word'] < 35]

def norm_val(instr):
    # Using the norm function defined earlier to remove $ and , symbols.
    # Not shown here but separate check confirms no decimal representations in string values.
    instr = norm_string(instr)
    try: 
        outint = int(instr)
    except:
        outint = 0
    return outint

jeopardy['clean_value'] = jeopardy['Value'].apply(norm_val)

# Straightforward transformation of airdate column
jeopardy['AirDate'] = pd.to_datetime(jeopardy['AirDate'])

# This function which will be applied over the rows of the dataset, doing all the processing, i.e. returning the percentage of
# answer words that are in the question.
def deducible(row):
    # Splitting the columns and initialising our counter
    split_answer = row['clean_answer'].split(' ')
    split_question = row['clean_question'].split(' ')
    match_count = 0
    
    # Remove the uninteresting words
    try:
        split_answer.remove('the')
    except:
        pass
    try:
        split_answer.remove('')
    except:
        pass
    
    # Null handling - we can't divide by zero so we'll return zero
    if len(split_answer) == 0:
        return 0
    
    #Check if answer words in question
    for sa in split_answer:
        if sa in split_question:
            match_count += 1
            
    #Return percentage of matched words
    return match_count / len(split_answer)

#Apply function over rows
jeopardy['answer_in_question'] = jeopardy.apply(deducible,axis=1)

print('Mean percentage of answer words in question: ' + str(round(jeopardy['answer_in_question'].mean(),2)))

# Stop word list, downloaded as csv from http://xpo6.com
import csv

f = open("C:\\Users\\User\\Documents\\Python projects\\stop-word-list.csv","r")
stop_words = list(csv.reader(f))
stop_words = [str.replace(c," ","") for c in stop_words[0]]
stop_words[:10]

jeopardy.sort_values('AirDate',ascending=True,inplace=True)

# Using a set for the previously used terms as we're only interested in unique terms
terms_used = set()

# Function performing the operation
def rep_q_check(row):
    split_question = row['clean_question'].split(' ')
    
    #reduce to 6 or more letter words and remove stopwords
    split_question = [s for s in split_question if (len(s) > 5) & (s not in stop_words)]
    
    #remove stopwords
    #split_question = [s for s in split_question if s not in stop_words]
    
    #de-dupe the list - note that the dataquest solution doesn't do this so I have slightly different result
    split_question = list(set(split_question))
    
    #create a counter and loop through word list
    match_count = 0
    for s in split_question:
        if s in terms_used:
            match_count += 1
        #now that we'have checked the word we can add it to the set
        terms_used.add(s)
    
    # Match count will still be zero if there were no 6 letter words, so null handling is simple:
    if len(split_question) > 0:
        match_count /= len(split_question)
    return match_count

jeopardy['question_overlap'] = jeopardy.apply(rep_q_check,axis=1)
jeopardy['question_overlap'].mean()

#We'll start by creating a high value column (1 if value over 800, 0 otherwise):
def value_assign(inint):
    if inint > 800:
        return 1
    else:
        return 0
jeopardy['high_value'] = jeopardy['clean_value'].apply(value_assign)

#Just resetting the index to make checking things a little easier
jeopardy.reset_index(inplace=True)
jeopardy.head()

#I'm going to time this as it seems slow and I want a comparison to other methods
import time
start = time.time()

#Next create a function that gets the count of high and low value questions containing a given word
def word_counter(instr):
    low_count = 0
    high_count = 0
    for index,row in jeopardy.iterrows():
        split_question = row['clean_question'].split(' ')
        if instr in split_question:
            if row['high_value'] == 1:
                high_count += 1
            else:
                low_count += 1
    return high_count, low_count

#The terms_used set contains all the words used in questions from previous steps
comparison_terms = list(terms_used)
comparison_terms = comparison_terms[0:100]

#Call the word counter and append results to a list
observed_expected = []
for ct in comparison_terms:
    observed_expected.append(word_counter(ct))
    
print(observed_expected[:10])

end = time.time()
print(end - start)

import time
start = time.time()

observed_results = {}
comp_terms = list(terms_used)

def word_counter2(row):
        
    split_question = row['clean_question'].split(' ')
    for ct in comp_terms:
        if ct in split_question:
            if ct in observed_results:
                new = observed_results[ct]
                if row['high_value'] == 1:
                    new = (new[0]+1,new[1])
                else:
                    new = (new[0],new[1]+1)
                observed_results[ct] = new
            else:
                if row['high_value'] == 1:
                    observed_results[ct] = (1,0)
                else:
                    observed_results[ct] = (0,1)
    return

for index,row in jeopardy.iterrows():
    word_counter2(row)

total_questions = jeopardy.shape[0]

end = time.time()
print(end - start)

# subsetting the observed results into a fresh dictionary
test_dict = {}
for o in observed_results:
    if (observed_results[o][0] > 4) & (observed_results[o][1] > 4): 
        test_dict[o] = observed_results[o]
        
len(test_dict)

#We will use a chisquared test to see if the observed frequencies of high and low value scores matches the expected
#Null hypothesis would be that the observed results are indistinguishable from chance 
#(i.e. that specific words aren't especially likely to be in high or low value questions)

from scipy.stats import chisquare
import numpy as np

#Now getting some expected values
high_value_count = jeopardy[jeopardy['high_value'] == 1].shape[0]
low_value_count = jeopardy.shape[0] - high_value_count

chi_squared = {}
expected = {}
for key in test_dict:
    #total is the number of occurances of the word we're looking at
    total = sum(test_dict[key])
    
    
    #proportion of rows containing the word
    total_prop = total / (high_value_count + low_value_count)
        
    #expected high and low results
    exp_high = total_prop * high_value_count
    exp_low = total_prop * low_value_count
    
    expected[key] = (round(exp_high), round(exp_low))
    
    obs = np.array([test_dict[key][0],test_dict[key][1]])
    exp = np.array([exp_high,exp_low])
    
    chi, p = chisquare(obs,exp)
    
    chi_squared[key] = [chi,p]

for r in chi_squared:
    if chi_squared[r][1] < 0.05:
        if test_dict[r][0] > expected[r][0]:
            print(r,' ',test_dict[r],' ',expected[r],' ',chi_squared[r])

