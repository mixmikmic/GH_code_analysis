# import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)
sns.set_style("whitegrid")

import sys

# allow plot display in notebook
get_ipython().magic('matplotlib inline')

# change default font/figure size for clearer viewing
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14

# define a list of column names (as strings for now - I have plans for time_stamp in future revisions)
col_names = ['time_stamp', 'raw_text', 'username']

# define the URL from which to retrieve the data (as a string)
url = 'https://raw.githubusercontent.com/analyticascent/stylext/master/feed_combo.csv'

# retrieve the CSV file and add the column names
df = pd.read_csv(url, header=0, names=col_names, na_filter=False)

df.head()

df.tail()

z = 0

for index, row in df.iterrows():
    z = z + 1
    print
    print
    print z, '-', row['time_stamp']
    print
    print row['raw_text']

# Check default limit - mine was 1000
sys.getrecursionlimit()

# Change default limit if necessary - I had to change mine to 2500
sys.setrecursionlimit(2000)

# Verify that change was successful
sys.getrecursionlimit()

# These are placeholders to be modified each time a new row is iterated
# z is the row number and total is the number of syllables counted overall
z = 0
total = 0

syllable = []

# for every row in the data frame...
for index, row in df.iterrows():
    # take the raw tweet text column
    word = str(row['raw_text'])
    # and convert that to lowercase
    word = word.lower()
    # add one to the previous value of z
    z = z + 1
    
    # syllables is another placeholder that will increase based on certain character combinations
    syllables = 0.0
    for i in range(len(word)) :
        
        # if the first letter in the word is a vowel then it's one syllable.
        
        if i == 0 and word[i] in "aeiouy" :
            syllables = syllables + 1
            
        # else if previous letter isn't a vowe
        elif word[i - 1] not in "aeiouy" :
                
            # if not the last letter and is a vowel
            if i < len(word) - 1 and word[i] in "aeiouy" :
                syllables = syllables + 1
                    
            # else if it is the last letter and it is a vowel that is not e.
            elif i == len(word) - 1 and word[i] in "aiouy" :
                syllables = syllables + 1

    # adjust syllables from 0 to 1.
    if len(word) > 0 and syllables == 0 :
        syllables == 0
        syllables = 1
    
    # this adds new syllable count of a given tweet to total from all previous ones
    total = total + syllables
    print
    # for every row, the tweet number 'z' will be printed, then the tweet syllable count, then total syllables accumulated
    print z, + syllables, + total
    
    syllable_count = int(syllables)
    syllable.append(syllable_count)

df['syllables'] = syllable   

# this gives us our final metrics
average_syllable = total_syllable / z
total_syllable = total

print
print
print "Average syllable count: ", average_syllable
print
print "Total syllable count: ", total_syllable

df.head()

df.dtypes

z = 0
total = 0

periods = []

for index, row in df.iterrows():
    tweet = str(row['raw_text'])
    z = z + 1
    total = total + tweet.count('.')
    print
    print z, tweet.count('.'), total
    
    period_count = tweet.count('.')
    period_count = int(period_count)
    periods.append(period_count)
    
df['periods'] = periods

# this gives us our final metrics
z = int(z)
average_period = total / z
total_period = total

print
print
print "Average period count: ", average_period
print
print "Total period count: ", total_period

df.head()

z = 0
total = 0

hyphens = []

for index, row in df.iterrows():
    tweet = str(row['raw_text'])
    z = z + 1
    total = total + tweet.count('-')
    print
    print z, tweet.count('-'), total
    
    hyphen_count = tweet.count('-')
    hyphen_count = int(hyphen_count)
    hyphens.append(hyphen_count)
    
df['hyphens'] = hyphens

# this gives us our final metrics
z = int(z)
average_hyphen = total / z
total_hyphen = total

print
print
print "Average hyphen count: ", average_hyphen
print
print "Total hyphen count: ", total_hyphen

df.head()

z = 0
total = 0

commas = []

for index, row in df.iterrows():
    tweet = str(row['raw_text'])
    z = z + 1
    total = total + tweet.count(',')
    print
    print z, tweet.count(','), total
    
    comma_count = tweet.count(',')
    comma_count = int(comma_count)
    commas.append(comma_count)
    
df['commas'] = commas

# this gives us our final metrics
z = int(z)
average_comma = total / z
total_comma = total

print
print
print "Average comma count: ", average_comma
print
print "Total comma count: ", total_comma

df.head()

z = 0
total = 0

semicolons = []

for index, row in df.iterrows():
    tweet = str(row['raw_text'])
    z = z + 1
    total = total + tweet.count(';')
    print
    print z, tweet.count(';'), total
    
    semicolon_count = tweet.count(';')
    semicolon_count = int(semicolon_count)
    semicolons.append(semicolon_count)
    
df['semicolons'] = semicolons

# this gives us our final metrics
z = int(z)
average_semicolon = total / z
total_semicolon = total

print
print
print "Average semicolon count: ", average_semicolon
print
print "Total semicolon count: ", total_semicolon

df.head()

z = 0
total = 0

exclamations = []

for index, row in df.iterrows():
    tweet = str(row['raw_text'])
    z = z + 1
    total = total + tweet.count('!')
    print
    print z, tweet.count('!'), total
    
    exclamation_count = tweet.count('!')
    exclamation_count = int(exclamation_count)
    exclamations.append(exclamation_count)
    
df['exclamations'] = exclamations

# this gives us our final metrics
z = int(z)
average_exclamation = total / z
total_exclamation = total

print
print
print "Average exclamation count: ", average_exclamation
print
print "Total exclamation count: ", total_exclamation

df.head()

z = 0
total = 0

questions = []

for index, row in df.iterrows():
    tweet = str(row['raw_text'])
    z = z + 1
    total = total + tweet.count('?')
    print
    print z, tweet.count('?'), total
    
    question_count = tweet.count('?')
    question_count = int(question_count)
    questions.append(question_count)
    
df['questions'] = questions

# this gives us our final metrics
z = int(z)
average_question = total / z
total_question = total

print
print
print "Average question count: ", average_question
print
print "Total question count: ", total_question

df.head()

z = 0
total = 0

quotes = []

for index, row in df.iterrows():
    tweet = str(row['raw_text'])
    z = z + 1
    total = total + tweet.count('"')
    print
    print z, tweet.count('"'), total
    
    quote_count = tweet.count('"')
    quote_count = int(quote_count)
    quotes.append(quote_count)
    
df['quotes'] = quotes

# this gives us our final metrics
z = int(z)
average_quote = total / z
total_quote = total

print
print
print "Average quote count: ", average_quote
print
print "Total quote count: ", total_quote

df.head()

z = 0
total = 0

dollars = []

for index, row in df.iterrows():
    tweet = str(row['raw_text'])
    z = z + 1
    total = total + tweet.count('$')
    print
    print z, tweet.count('$'), total
    
    dollar_count = tweet.count('$')
    dollar_count = int(dollar_count)
    dollars.append(dollar_count)
    
df['dollars'] = dollars

# this gives us our final metrics
z = int(z)
average_dollar = total / z
total_dollar = total

print
print
print "Average dollar count: ", average_dollar
print
print "Total dollar count: ", total_dollar

df.head()

z = 0
total = 0

percentages = []

for index, row in df.iterrows():
    tweet = str(row['raw_text'])
    z = z + 1
    total = total + tweet.count('%')
    print
    print z, tweet.count('%'), total
    
    percentage_count = tweet.count('%')
    percentage_count = int(percentage_count)
    percentages.append(percentage_count)
    
df['percentages'] = percentages

# this gives us our final metrics
z = int(z)
average_percentage = total / z
total_percentage = total

print
print
print "Average percentage count: ", average_percentage
print
print "Total percentage count: ", total_percentage

df.head()

z = 0
total = 0

ands = []

for index, row in df.iterrows():
    tweet = str(row['raw_text'])
    z = z + 1
    total = total + tweet.count('&')
    print
    print z, tweet.count('&'), total
    
    and_count = tweet.count('&')
    and_count = int(percentage_count)
    ands.append(and_count)
    
df['ands'] = ands

# this gives us our final metrics
z = int(z)
average_and = total / z
total_and = total

print
print
print "Average and count: ", average_and
print
print "Total and count: ", total_and

df.head()

z = 0
total = 0

asterisks = []

for index, row in df.iterrows():
    tweet = str(row['raw_text'])
    z = z + 1
    total = total + tweet.count('*')
    print
    print z, tweet.count('*'), total
    
    asterisk_count = tweet.count('*')
    asterisk_count = int(asterisk_count)
    asterisks.append(asterisk_count)
    
df['asterisks'] = asterisks

# this gives us our final metrics
z = int(z)
average_asterisk = total / z
total_asterisk = total

print
print
print "Average asterisk count: ", average_asterisk
print
print "Total asterisk count: ", total_asterisk

df.head()

z = 0
total = 0

pluses = []

for index, row in df.iterrows():
    tweet = str(row['raw_text'])
    z = z + 1
    total = total + tweet.count('+')
    print
    print z, tweet.count('+'), total
    
    plus_count = tweet.count('+')
    plus_count = int(plus_count)
    pluses.append(plus_count)
    
df['pluses'] = pluses

# this gives us our final metrics
z = int(z)
average_plus = total / z
total_plus = total

print
print
print "Average plus count: ", average_plus
print
print "Total plus count: ", total_plus

df.head()

z = 0
total = 0

equals = []

for index, row in df.iterrows():
    tweet = str(row['raw_text'])
    z = z + 1
    total = total + tweet.count('=')
    print
    print z, tweet.count('='), total
    
    equal_count = tweet.count('=')
    equal_count = int(equal_count)
    equals.append(equal_count)
    
df['equals'] = equals

# this gives us our final metrics
z = int(z)
average_equal = total / z
total_equal = total

print
print
print "Average equal count: ", average_equal
print
print "Total equal count: ", total_equal

df.head()

z = 0
total = 0

slashes = []

for index, row in df.iterrows():
    tweet = str(row['raw_text'])
    z = z + 1
    total = total + tweet.count('/')
    print
    print z, tweet.count('/'), total
    
    slash_count = tweet.count('/')
    slash_count = int(slash_count)
    slashes.append(slash_count)
    
df['slashes'] = slashes

# this gives us our final metrics
z = int(z)
average_slash = total / z
total_slash = total

print
print
print "Average slash count: ", average_slash
print
print "Total slash count: ", total_slash

df.head()

z = 0
total = 0

hashes = []

for index, row in df.iterrows():
    tweet = str(row['raw_text'])
    z = z + 1
    total = total + tweet.count('#')
    print
    print z, tweet.count('#'), total
    
    hash_count = tweet.count('#')
    hash_count = int(hash_count)
    hashes.append(hash_count)
    
df['hashes'] = hashes

# this gives us our final metrics
z = int(z)
average_hashtag = total / z
total_hashtag = total

print
print
print "Average hashtag count: ", average_hashtag
print
print "Total hashtag count: ", total_hashtag

df.head()

z = 0
total = 0

ats = []

for index, row in df.iterrows():
    tweet = str(row['raw_text'])
    z = z + 1
    total = total + tweet.count('@')
    print
    print z, tweet.count('@'), total
    
    at_count = tweet.count('@')
    at_count = int(at_count)
    ats.append(at_count)
    
df['replies'] = ats

# this gives us our final metrics
z = int(z)
average_reply = total / z
total_reply = total

print
print
print "Average reply count: ", average_reply
print
print "Total reply count: ", total_reply

df.head()

z = 0
total = 0

retweets = []

for index, row in df.iterrows():
    tweet = str(row['raw_text'])
    z = z + 1
    total = total + tweet.count('RT')
    print
    print z, tweet.count('RT'), total
    
    retweet_count = tweet.count('RT')
    retweet_count = int(retweet_count)
    retweets.append(retweet_count)
    
df['retweets'] = retweets

# this gives us our final metrics
z = int(z)
average_retweet = total / z
total_retweet = total

print
print
print "Average retweet count: ", average_retweet
print
print "Total retweet count: ", total_retweet

df.head()

z = 0
total = 0

links = []

for index, row in df.iterrows():
    tweet = str(row['raw_text'])
    z = z + 1
    total = total + tweet.count('http')
    print
    print z, tweet.count('http'), total
    
    link_count = tweet.count('http')
    link_count = int(link_count)
    links.append(link_count)
    
df['links'] = links

# this gives us our final metrics
z = int(z)
average_link = total / z
total_link = total

print
print
print "Average link count: ", average_link
print
print "Total link count: ", total_link

df.head()

z = 0
total = 0

smiles = []

for index, row in df.iterrows():
    tweet = str(row['raw_text'])
    z = z + 1
    total = total + tweet.count(':-)')
    print
    print z, tweet.count(':-)'), total
    
    smile_count = tweet.count(':-)')
    smile_count = int(smile_count)
    smiles.append(smile_count)
    
df['smiles'] = smiles

# this gives us our final metrics
z = int(z)
average_smile = total / z
total_smile = total

print
print
print "Average smile count: ", average_smile
print
print "Total smile count: ", total_smile

df.head()

z = 0
total = 0

bigsmiles = []

for index, row in df.iterrows():
    tweet = str(row['raw_text'])
    z = z + 1
    total = total + tweet.count(':-D')
    print
    print z, tweet.count(':-D'), total
    
    bigsmile_count = tweet.count(':-D')
    bigsmile_count = int(bigsmile_count)
    bigsmiles.append(bigsmile_count)
    
df['bigsmiles'] = bigsmiles

# this gives us our final metrics
z = int(z)
average_bigsmile = total / z
total_bigsmile = total

print
print
print "Average bigsmile count: ", average_bigsmile
print
print "Total bigsmile count: ", total_bigsmile

df.head()

z = 0
total = 0

winks = []

for index, row in df.iterrows():
    tweet = str(row['raw_text'])
    z = z + 1
    total = total + tweet.count(';-)')
    print
    print z, tweet.count(';-)'), total
    
    wink_count = tweet.count(';-)')
    wink_count = int(wink_count)
    winks.append(wink_count)
    
df['winks'] = winks

# this gives us our final metrics
z = int(z)
average_wink = total / z
total_wink = total

print
print
print "Average wink count: ", average_wink
print
print "Total wink count: ", total_wink

df.head()

z = 0
total = 0

bigwinks = []

for index, row in df.iterrows():
    tweet = str(row['raw_text'])
    z = z + 1
    total = total + tweet.count(';-D')
    print
    print z, tweet.count(';-D'), total
    
    bigwink_count = tweet.count(';-D')
    bigwink_count = int(bigwink_count)
    bigwinks.append(bigwink_count)
    
df['bigwinks'] = bigwinks

# this gives us our final metrics
z = int(z)
average_bigwink = total / z
total_bigwink = total

print
print
print "Average bigwink count: ", average_bigwink
print
print "Total bigwink count: ", total_bigwink

df.head()

z = 0
total = 0

unsures = []

for index, row in df.iterrows():
    tweet = str(row['raw_text'])
    z = z + 1
    total = total + tweet.count(':-/')
    print
    print z, tweet.count(':-/'), total
    
    unsure_count = tweet.count(':-/')
    unsure_count = int(unsure_count)
    unsures.append(unsure_count)
    
df['unsures'] = unsures

# this gives us our final metrics
z = int(z)
average_unsure = total / z
total_unsure = total

print
print
print "Average unsure count: ", average_unsure
print
print "Total unsure count: ", total_unsure

df.head()

mapping = {'DLin71_feed': 1, 'NinjaEconomics': 2}

df.replace({'username': mapping})

df.boxplot(column='syllables', by='username')

df.boxplot(column='periods', by='username')

df.boxplot(column='hyphens', by='username')

df.boxplot(column='commas', by='username')

df.boxplot(column='semicolons', by='username')

df.boxplot(column='exclamations', by='username')

df.boxplot(column='questions', by='username')

df.boxplot(column='quotes', by='username')

df.boxplot(column='dollars', by='username')

df.boxplot(column='percentages', by='username')

df.boxplot(column='ands', by='username')

df.boxplot(column='asterisks', by='username')

df.boxplot(column='pluses', by='username')

df.boxplot(column='equals', by='username')

df.boxplot(column='slashes', by='username')

df.boxplot(column='hashtags', by='username')

df.boxplot(column='replies', by='username')

df.boxplot(column='retweets', by='username')

df.boxplot(column='links', by='username')

df.boxplot(column='smiles', by='username')

df.boxplot(column='bigsmiles', by='username')

df.boxplot(column='winks', by='username')

df.boxplot(column='bigwinks', by='username')

df.boxplot(column='unsures', by='username')

df.to_csv('post_feed.csv', encoding='utf-8')

