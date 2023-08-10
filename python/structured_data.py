import os
import re
import sys
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
 
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

sys.path.append('./auxiliary files')

from title_detection import *
from detect_ending import *

text = open('Snippet_Globe_displayad_19790114_226.txt').read()
page_identifier = 'Globe_displayad_19790114_226'
print(text) # posting text

# remove emypty lines
text_by_line = [w for w in re.split('\n',text) if not w=='']

# reset lines (see title_detection.py)
text_reset_line = CombineUppercase(text_by_line)
text_reset_line = UppercaseNewline(text_reset_line,'\n') #assign new line when an uppercase word is found
text_reset_line = CombineUppercase(text_reset_line) #re-combine uppercase words together

# remove extra white spaces
text_reset_line = [' '.join([y for y in re.split(' ',w) if not y=='']) for w in text_reset_line]
# remove empty lines
text_reset_line = [w for w in text_reset_line if not w=='']

# print results
text_reset_line

# define indicators if job title detected
title_found = '---titlefound---'

# list of job title personal nouns
TitleBaseFile = open('./auxiliary files/TitleBase.txt').read()
TitleBaseList = [w for w in re.split('\n',TitleBaseFile) if not w=='']
print('--- Examples of job title personal nouns ---')
print(TitleBaseList[:15]) 

text_detect_title = ['']*len(text_reset_line)
PreviousLineIsUppercaseTitle = False

# assign a flag of '---titlefound---' to lines where we detect a job title

for i in range(0,len(text_reset_line)):
    line = text_reset_line[i]
    line_no_hyphen = re.sub('-',' ',line.lower())
    tokens = word_tokenize(line_no_hyphen)
    
    Match = list(set(tokens).intersection(TitleBaseList)) # see if the line has words in TitleBaseList 
        
    if Match and DetermineUppercase(line): # uppercase job title
        text_detect_title[i] = ' '.join([w for w in re.split(' ',line) if not w=='']) + title_found
        # adding a flag that a title is found
        # ' '.join([w for w in split(' ',line) if not w=='']) is to remove extra spaces from 'line'
        PreviousLineIsUppercaseTitle = True
    elif Match and len(tokens) <= 2:
        # This line allows non-uppercase job titles
        # It has to be short enough => less than or equal to 2 words.
        # In addition, the previous line must NOT be a uppercase job title. 
        if PreviousLineIsUppercaseTitle == False:
            text_detect_title[i] = ' '.join([w for w in re.split(' ',line) if not w=='']) + title_found
            PreviousLineIsUppercaseTitle = False
        else:
            text_detect_title[i] = ' '.join([w for w in re.split(' ',line) if not w==''])
            PreviousLineIsUppercaseTitle = False
    else:
        text_detect_title[i] = ' '.join([w for w in re.split(' ',line) if not w==''])
        PreviousLineIsUppercaseTitle = False

[w for w in text_detect_title if re.findall(title_found,w)]

ending_found = '---endingfound---'
text_assign_flag = list()

# see "detect_ending.py"

for line in text_detect_title:
    AddressFound , EndingPhraseFound = AssignFlag(line)
    if AddressFound == True or EndingPhraseFound == True:
        text_assign_flag.append(line + ending_found)
    else:
        text_assign_flag.append(line)

[w for w in text_assign_flag if re.findall(ending_found,w)]

text_assign_flag

split_indicator = '---splithere---'
split_by_title = list() 
split_posting = list()

# -----split if title is found-----

for line in text_assign_flag:
    if re.findall(title_found,line):
        #add a split indicator BEFORE the line with title 
        split_by_title.append(split_indicator + '\n' + line)
    else:
        split_by_title.append(line) # if not found, just append the line back in 
            
split_by_title = [w for w in re.split('\n','\n'.join(split_by_title)) if not w=='']

# -----split if any ending phrase and/or address is found-----

for line in split_by_title:
    line_remove_ending_found = re.sub(ending_found,'',line) #remove the ending flag
    if re.findall(ending_found,line):
        #add a split indicator AFTER the line where the pattern is found
        split_posting.append( line_remove_ending_found + '\n' + split_indicator)
    else:
        split_posting.append( line_remove_ending_found ) # if not found, just append the line back in 

# after assigning the split indicators, we can use python command to split the ads.        
split_posting = [w for w in re.split(split_indicator,'\n'.join(split_posting)) if not w=='']

for ad in split_posting:
    print(re.sub('\n','',ad)) #print out each ad, ignoring the line break indicators. 
    print('---splithere---')

all_flag = re.compile('|'.join([title_found,ending_found]))

num_ad = 0 #initialize ad number within displayad

final_output = list()

for ad in split_posting:
    
    ad_split_line = [w for w in re.split('\n',ad) if not w=='']
        
    # --------- record title ----------

    title_this_ad = [w for w in ad_split_line if re.findall(title_found,w)] 
    #see if any line is a title
            
    if len(title_this_ad) == 1: #if we do have a title
        title_clean = re.sub(all_flag,'',title_this_ad[0].lower()) 
        #take out the flags and revert to lowercase

        title_clean = ' '.join([y for y in re.split(' ',title_clean) if not y==''])
    else:
        title_clean = ''

    # --------- record content ----------
        
    ad_content = [w for w in ad_split_line if not re.findall(title_found,w)] # take out lines with title
    ad_content = ' '.join([w for w in ad_content if not w==''])
    #delete empty lines + combine all the line together (within an ad)
        
    ad_content = re.sub(all_flag,'',ad_content) 
    #take out all the flags

    # --------- record output ----------

    num_ad += 1
    output = [str(page_identifier),str(num_ad),str(title_clean),str(ad_content)]    
    final_output.append( '|'.join(output) )

# final output     
final_output_file = open('structured_data.txt','w')
final_output_file.write('\n'.join(final_output))
final_output_file.close()

# print out final output
structured_posting = open('structured_data.txt').read()
structured_posting = re.split('\n',structured_posting)
for ad in structured_posting:
    print(ad)

