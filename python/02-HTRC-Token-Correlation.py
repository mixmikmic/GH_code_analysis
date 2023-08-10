get_ipython().run_cell_magic('capture', '', '!pip install nltk')

from htrc_features import FeatureReader
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

get_ipython().system('rm -rf local-folder')
download_output = get_ipython().getoutput('htid2rsync --f novels-word-use.txt | rsync -azv --files-from=- data.sharc.hathitrust.org::features/ local-folder/')
suffix = '.json.bz2'
file_paths = ['local-folder/' + path for path in download_output if path.endswith(suffix)]

fr_novels = FeatureReader(file_paths)
for vol in fr_novels:
    print(vol.title)

title_word = 'mockingbird'

for vol in fr_novels:
    if title_word.lower() in vol.title.lower():
        print('Current volume:', vol.title)
        break

tokens = vol.tokenlist(pos=False, case=False, pages=False).sort_values('count', ascending=False)

freqs = []
for count in tokens['count']:
    freqs.append(count/sum(tokens['count']))
    
tokens['rel_frequency'] = freqs
tokens

get_ipython().magic('matplotlib inline')
# Build a list of frequencies and a list of tokens.
freqs_1, tokens_1 = [], []
for i in range(15):  # top 8 words
    freqs_1.append(freqs[i])
    tokens_1.append(tokens.index.get_level_values('lowercase')[i])

# Create a range for the x-axis
x_ticks = np.arange(len(tokens_1))

# Plot!
plt.bar(x_ticks, freqs_1)
plt.xticks(x_ticks, tokens_1, rotation=90)
plt.ylabel('Frequency', fontsize=14)
plt.xlabel('Token', fontsize=14)
plt.title('Common token frequencies in "' + vol.title[:14] + '..."', fontsize=14)

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from string import punctuation

print(stopwords.words('english'))

print()

print(punctuation)

freqs_filtered, tokens_filtered, i = [], [], 0
while len(tokens_filtered) < 10:
    if tokens.index.get_level_values('lowercase')[i] not in stopwords.words('english') + list(punctuation):
        freqs_filtered.append(freqs[i])
        tokens_filtered.append(tokens.index.get_level_values('lowercase')[i])
    i += 1

# Create a range for the x-axis
x_ticks = np.arange(len(freqs_filtered))

# Plot!
plt.bar(x_ticks, freqs_filtered)
plt.xticks(x_ticks, tokens_filtered, rotation=90)
plt.ylabel('Frequency', fontsize=14)
plt.xlabel('Token', fontsize=14)
plt.title('Common token frequencies in "' + vol.title[:14] + '..."', fontsize=14)

# A function to return the most common noun of length at least word_length in the volume.
# NOTE: word_length defaults to 2.
# e.g. most_common_noun(fr_novels.first) returns 'time'.

def most_common_noun(vol, word_length=2):   
    # Build a table of common nouns
    tokens_1 = vol.tokenlist(pages=False, case=False)
    nouns_only = tokens_1.loc[(slice(None), slice(None), ['NN']),]
    top_nouns = nouns_only.sort_values('count', ascending=False)

    token_index = top_nouns.index.get_level_values('lowercase')
    
    # Choose the first token at least as long as word_length with non-alphabetical characters
    for i in range(max(token_index.shape)):
        if (len(token_index[i]) >= word_length):
            if("'", "!", ",", "?" not in token_index[i]):
                return token_index[i]
    print('There is no noun of this length')
    return None

most_common_noun(vol, 15)

# Return the usage frequency of a given word in a given volume. 
# NOTE: frequency() returns a dictionary entry of the form {'word': frequency}.
# e.g. frequency(fr_novels.first(), 'blue') returns {'blue': 0.00012}

def frequency(vol, word):
    t1 = vol.tokenlist(pages=False, pos=False, case=False)
    token_index = t1[t1.index.get_level_values("lowercase") == word]
    
    if len(token_index['count'])==0:
        return {word: 0}
    
    count = token_index['count'][0]
    freq = count/sum(t1['count'])
    
    return {word: float('%.5f' % freq)}

frequency(vol, 'blue')

# Returns a plot of the usage frequencies of the given word across all volumes in the given FeatureReader collection.
# NOTE: frequencies are given as percentages rather than true ratios.
def frequency_bar_plot(word, fr):
    freqs, titles = [], []
    for vol in fr:
        title = vol.title
        short_title = title[:6] + (title[6:] and '..')
        freqs.append(100*frequency(vol, word)[word])
        titles.append(short_title)
        
    # Format and plot the data
    x_ticks = np.arange(len(titles))
    plt.bar(x_ticks, freqs)
    plt.xticks(x_ticks, titles, fontsize=10, rotation=45)
    plt.ylabel('Frequency (%)', fontsize=12)
    plt.title('Frequency of "' + word + '"', fontsize=14)

frequency_bar_plot('blue', fr_novels)

# Use the provided frequency functions to plot the most common 5-letter noun in "To Kill a Mockinbird".
# Your solution should be just one line of code.




download_output = get_ipython().getoutput('htid2rsync --f math-collection.txt | rsync -azv --files-from=- data.sharc.hathitrust.org::features/ local-folder/')
file_paths = ['local-folder/' + path for path in download_output if path.endswith(suffix)]
fr_math = FeatureReader(file_paths)

fr_math = FeatureReader(file_paths)
for vol in fr_math:
    print(vol.title)

# Returns a DF of relative frequencies, volume years, and page counts, along with a scatter plot.
# NOTE: frequencies are given in percentages rather than true ratios.
def frequency_by_year(query_word, fr):
    volumes = pd.DataFrame()
    years, page_counts, query_freqs = [], [], []

    for vol in fr:
        years.append(int(vol.year))
        page_counts.append(int(vol.page_count))
        query_freqs.append(100*frequency(vol, query_word)[query_word])
    
    volumes['year'], volumes['pages'], volumes['freq'] = years, page_counts, query_freqs
    volumes = volumes.sort_values('year')
    
    # Set plot dimensions and labels
    scatter_plot = volumes.plot.scatter('year', 'freq', color='black', s=50, fontsize=12)
    plt.ylim(0-np.mean(query_freqs), max(query_freqs)+np.mean(query_freqs))
    plt.ylabel('Frequency (%)', fontsize=12)
    plt.xlabel('Year', fontsize=12)
    plt.title('Frequency of "' + query_word + '"', fontsize=14)
    
    return volumes.head(10)

frequency_by_year('quadratic', fr_math)

frequency_by_year('prime', fr_math)

frequency_by_year('factor', fr_math)



