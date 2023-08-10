import sys
import unicodedata
import csv
import got
from IPython.core.display import HTML
    
# Setting up 
KEYWORD = "forest restoration" # Sets the keyword to look for in the tweet
START_DATE = "2017-07-01" # Sets the start date for the search (must be in YYYY-MM-DD format as a string)
END_DATE = "2017-07-10" # Sets the end date for the search
tweetCriteria = got.manager.TweetCriteria().setQuerySearch(KEYWORD).setSince(START_DATE).setUntil(END_DATE).setMaxTweets(5)
results = got.manager.TweetManager.getTweets(tweetCriteria)

for i in range(len(results)):
    print results[i]
    

# Creates an empty source and target list
source = []
target = []
# Loops over each result to parse the data stored in the result
for i in range(len(results)):
    # Stores each result in a temporary variable called "tweet"
    tweet = results[i]
    # Stores the username of the creator of content in "username" as a string 
    username = tweet.username
    # Stores the list of mentions in the tweet in "mentions" as a string
    mentions = tweet.mentions
    # Accounts for the case where no other user is mentioned in the tweet
    # Stores the username of the creator of content in both the source and target lists
    if "@" not in mentions:
        source.append(username)
        target.append(username)
    # Counts the number of other accounts mentioned in the tweet using the "@" character
    # For each mention, the username and the specific account mentioned are appended to the source and target lists respectively
    else:
        number_of_mentions = mentions.count("@") + 1
        split_mentions = mentions.split("@")
        for i in range(number_of_mentions):
            if i > 0:
                source.append(username)
                target.append(split_mentions[i])
                

# Specifies the filename to store the source and target columns (REMEMBER TO ADD .csv AT THE END OF THE FILENAME)
filename = "ForestRestorationTweets.csv"
with open(filename, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    # Writes the source and target lists for each row in the CSV file
    for i in range(len(source)):
        text = source[i],target[i]
        writer.writerow(text)

get_ipython().magic('reset')

import sys
import unicodedata
import csv
import got
from IPython.core.display import HTML
    
# Setting up 
USERNAME = "restoreforward"
START_DATE = "2017-06-01"
END_DATE = "2017-07-10"
# Sets the parameters of the search through the TweetCriteria class in got.manager
# List of TweetCriteria parameters:
# setUsername, setSince, setUntil, setQuerySearch, setMaxTweets, setTopTweets, setNear, setWithin
tweetCriteria = got.manager.TweetCriteria().setUsername(USERNAME).setSince(START_DATE).setUntil(END_DATE)
results = got.manager.TweetManager.getTweets(tweetCriteria)
# Loops through each result to see each account mentioned in each tweet by "restoreforward"
for i in range(len(results)):
    if results[i].mentions != "":
        print results[i].mentions

# Creates an empty list to store tweet text
tweet_text = []
# Loops through each result to append the tweet text to the tweet text list
for i in range(len(results)):
    tweet_text.append(results[i].text)

def printTweetText(text_file):
    # Writes each element in the tweet text list to a text file 
    # REMEMBER TO ADD .txt TO THE END OF THE TEXT FILE NAME (eg. "RestorationTweets.txt")
    with open(text_file, 'wb') as text_file:
        for i in range(len(tweet_text)):
            text = tweet_text[i].encode(sys.stdout.encoding, errors='replace')
            text = ' ' + text
            text_file.write(text)

def get_text(filename): # Remember to including .txt at the end of filename
    text_file = open(filename, 'r')
    lines = text_file.read().split(' ') # Splits the text by a space
    return lines

def get_frequencies(lines):
    # Counts the number of words in the text file
    array_lines = range(len(lines))
    # Creates an empty list to store word counts
    word_array = []
    # Loops over each word in the text file and determines whether it is already in the word array
    # If it is not in the word array, it adds the word, with a word count of "1" to the word array
    # If it is already in the word array, it increases the word count by 1
    for i in array_lines:
        if lines[i] not in [word[0] for word in word_array]:
            word_array.append([lines[i],1])
        else:
            word_array_length = range(len(word_array))
            for j in word_array_length:
                if word_array[j][0] == lines[i]:
                    word_array[j][1] += 1
    # Sorts the word array by word count, and returns the word count in descending order (most frequent first)                
    word_array = sorted(word_array, key=lambda x: int(x[1]))
    word_array = list(reversed(word_array))
    
    return word_array

def filter_array(word_array):
    # Counts the number of distinct words in the word array
    length_array = range(len(word_array))
    # Creates a list of words that are "fluff" - words you don't want inlcuded in the analysis
    fluff = ["ENTER WHAT WORDS YOU WANT TO FILTER HERE"]
    # EXAMPLE:
    # fluff = ['and','the','because','for','she','he','that','have','not']
    # Loops over each word in the word array and kills the word counts for words that are either:
    # (a) less than 3 letters long
    # (b) in the fluff list
    for i in length_array:
        if len(word_array[i][0]) < 3:
            word_array[i][1] = 0
        if word_array[i][0] in fluff:
            word_array[i][1] = 0
    # Sorts the newly filtered word array by word count, and returns the word count in descending order (most frequent first)        
    word_array = sorted(word_array, key=lambda x: int(x[1]))
    word_array = list(reversed(word_array))
    
    return word_array



