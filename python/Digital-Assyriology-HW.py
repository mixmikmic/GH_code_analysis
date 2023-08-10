get_ipython().magic('pylab inline')
from datascience import *
import pandas as pd
import numpy as np
import os
import re

def term_finder (line):
    #returns the meanings of words in a line of text
    terms = re.findall(r"(?<=\[)(.*?)(?=\])", line)
    return terms

def proper_noun_finder(line):
    #returns a list of all the proper nouns in a line of text
    nouns = re.findall(r"(?<=\:)(.*?)(?=\[)", line)
    nouns = [word for word in nouns if (len(word) > 1 and word[0].isupper() and not word[1].isupper())]
    return nouns

def speech_article_finder(line, proper_noun_filter = True):
    #returns the speech articles for proper_nouns or all words
    terms = re.findall(r"(?<=\])(.*?)(?=\s)", line)
    if proper_noun_filter:
        articles = [term for term in terms if term in proper_nouns]
    else: 
        articles = terms
    return articles

proper_nouns = {  #All Proper nouns (things marked with a [1])
'CN': 'Constellation Name (star)',
'DN': 'Deity Name',
'EN': 'Ethnicity Name',
'FN': 'Field Name',
'GN': 'Geographical Name (for regions and countries)',
'MN': 'Month Name',
'ON': 'Object Name (usually for objects associated with a god)',
'PN': 'Personal Name',
'RN': 'Royal Name',
'SN': 'Settlement Name',
'TN': 'Temple Name',
'WN': 'Water Name',
}

simple_terms = {  #All other terms that we've seen but that are not names.
'AJ': 'Adjective',
'AV': 'Adverb',
'C': 'Conjunction',
'N': 'Noun',
'NU': 'Number',
'PD': 'Part of Speech',
'V': 'Verb',
}

text_table = Table.read_table('Enmerkar.txt', sep = ',') ### NEED FILES NAMES between folders
#read in the table

text_table = text_table.with_columns([
    'terms', text_table.apply(term_finder, 'text'), 
    'proper_nouns', text_table.apply(proper_noun_finder, 'text'), 
    'speech_articles', text_table.apply(speech_article_finder, 'text')
    ])
#clean it up with the term finders 

text_table = text_table.drop(['etcsl_no', 'text_name'])
text_table #remove the redundant information

def partitioning(line_no):    #Partitioning depends on the text used.
    
    ln = int(''.join(c for c in line_no if c.isdigit()))

    if(ln <= 13):
        return "1.1"
    elif (ln <= 21):
        return "1.2"
    
    elif (ln <= 39):
        return "2.1.1"
    elif (ln <= 51):
        return "2.1.2"
    elif (ln <= 69):
        return "2.1.3"
    
    elif (ln <= 76):
        return "2.2.1"
    elif (ln <= 90):
        return "2.2.2"
    elif (ln <= 113):
        return "2.2.3"
    
    elif (ln <= 127):
        return "2.3.1"
    elif (ln <= 132):
        return "2.3.2"
    elif (ln <= 134):
        return "2.3.3"
    
    elif (ln <= 138):
        return "3.1.1"
    elif (ln <= 149):
        return "3.1.2"
    elif (ln <= 162):
        return "3.1.3"
    elif (ln <= 169):
        return "3.1.4"
    
    elif (ln <= 184):
        return "3.2.1"
    elif (ln <= 197):
        return "3.2.2"
    elif (ln <= 205):
        return "3.2.3"
    elif (ln <= 210):
        return "3.2.4"
    elif (ln <= 221):
        return "3.2.5"
    
    elif (ln <= 227):
        return "4.1"
    
    elif (ln <= 248):
        return "4.2.1"
    elif (ln <= 254):
        return "4.2.2"
    elif (ln <= 263):
        return "4.2.3"
    elif (ln <= 273):
        return "4.2.4"
    
    elif (ln <= 280):
        return "5.1"
    elif (ln <= 283):
        return "5.2"
    elif (ln <= 310):
        return "B"
    return "0"

def small_partition(line_no):
    ln = int(''.join(c for c in line_no if c.isdigit()))
    if(ln <= 13):
        return "1.1"
    elif (ln <= 21):
        return "1.2"
    elif (ln <= 69):
        return "2.1"
    elif (ln <= 113):
        return "2.2"
    elif (ln <= 134):
        return "2.3"
    elif (ln <= 169):
        return "3.1"
    elif (ln <= 221):
        return "3.2"
    elif (ln <= 227):
        return "4.1"
    elif (ln <= 273):
        return "4.2"
    elif (ln <= 280):
        return "5.1"
    elif (ln <= 283):
        return "5.2"
    elif (ln <= 310):
        return "6"
    return "0"

text_table.append_column('section', text_table.apply(small_partition, 'l_no'))
text_graph = text_table.select(['proper_nouns', 'speech_articles', 'section']).group('section', list) 

def list_flattening(pn_list):  #define the list flattening function
    return [noun for nouns in pn_list for noun in nouns]

#flatten the lists that we're concerned with
text_graph.append_column('speech articles', text_graph.apply(list_flattening, 'speech_articles list'))
text_graph.append_column('proper nouns', text_graph.apply(list_flattening, 'proper_nouns list'))

#drop the columns we don't need
text_graph = text_graph.drop(['proper_nouns list', 'speech_articles list'])

text_graph

def partitioner (i):
    rows = []
    section = text_graph['section'][i]
    speech_articles = text_graph['speech articles'][i]
    proper_nouns = text_graph['proper nouns'][i]
    for j in range(len(speech_articles)):
        article = speech_articles[j]
        proper_noun = proper_nouns[j]
        rows.append([section, article, proper_noun])
    return rows

text_table_section = Table(['section', 'speech articles', 'proper nouns'])
for i in range(text_graph.num_rows):
    text_table_section = text_table_section.with_rows(partitioner(i))
text_table_section

proper_noun_by_section = text_table_section.pivot('proper nouns', rows = 'section')
name_counts = []
for name in proper_noun_by_section.drop('section').labels:
    name_counts.append([name, np.sum(proper_noun_by_section[name])])

top_7_names = ['FILL THIS IN WITH THE TOP 7 names']
name_counts

names_graph = proper_noun_by_section.with_column(
    'section', range(1, proper_noun_by_section.num_rows+1))

top_7_names_graph = names_graph.select(['Aratta', 'Unug', 'section']).plot('section')
#notice Aratta is the only one mentioned in the section 4.2.3

names_graph.plot('section') # see who is mentioned the most



