import numpy as np
import pandas as pd

from text_preprocess import remove_symbol, remove_stopword

np.random.seed(7) # Fix random seed for reproducibility

"""
_____________________
Step 1: Load articles
_____________________
"""
# The data is mined from coindesk
data = pd.read_json('data/manualVerified_senti_2600.json')


"""
_________________________
Step 2: Load dictionaries
_________________________
"""

dict_1 = pd.read_excel('data/WordDatabase/1/LoughranMcDonald_MasterDictionary_2014.xlsx')

poswd_1 = [x['Word'].lower() for i,x in dict_1.iterrows() if x['Positive']]

# str() is necessary to use .lower() on the word 'False'
# Otherwise 'False' will be treated as boolean and gives error with .lower()
negwd_1 = [str(x['Word']).lower() for i,x in dict_1.iterrows() if x['Negative']]

poswd_1 = pd.DataFrame(columns=['Positive'], data=poswd_1)
negwd_1 = pd.DataFrame(columns=['Negative'], data=negwd_1)


# Load positive and negative words from 2nd dictionary source
poswd_2 = pd.read_csv('data/WordDatabase/2/positive-words.txt', 
                      sep=" ", comment=';', names=['Positive']) 

# encoding is required here since some words have '-'
negwd_2 = pd.read_csv('data/WordDatabase/2/negative-words.txt', 
                      sep=" ", comment=';', names=['Negative'], encoding='latin-1') 



"""
_________________________
Step 3: Clean the data
_________________________
"""

# Remove \xa0
data['contents'] = data['contents'].apply(lambda x: x.replace(u'\xa0', u' '))

# Remove text after "The leader in blockchain news"
data['contents'] = data['contents'].apply(lambda x: x.split('The leader in blockchain news')[0])
data['contents'] = data['contents'].apply(lambda x: x.split('Disclosure:')[0])
data['contents'] = data['contents'].apply(lambda x: x.split('Disclaimer:')[0])

# Remove punctuations and stopwords (a, the, is ...)
data['contents clean'] = data['contents'].apply(remove_symbol).apply(remove_stopword)

# Tokeniz the sentence into a list of words
data['contents tokens'] = data['contents clean'].apply(lambda x: x.split())



"""
_______________________________________________________________
Step 4: Determine sentiment based on two different dictionaries
_______________________________________________________________
"""

# Define functions to measure the sentiment
def count_words(txt, diction):
    # Function to count words in "txt" that are in "diction"
    # CAUTION: 'diction' should be a list, not a pandas dataframe
    # "if word in diction" works ONLY for a list
    wrds = [word for word in txt if word in diction]
    return len(wrds)

def measure_senti(txt, neg_dict, pos_dict):
    return(count_words(txt,pos_dict) - count_words(txt,neg_dict))


# Convert negwd_2['Negative'] to a list. Doesn't work otherwise
data['Senti 1'] = data['contents tokens'].apply(
    lambda x: measure_senti(x,negwd_1['Negative'].tolist(), poswd_1['Positive'].tolist()))

data['Senti 2'] = data['contents tokens'].apply(
    lambda x: measure_senti(x,negwd_2['Negative'].tolist(), poswd_2['Positive'].tolist()))



"""
_____________________________________
Step 5: Determine the final sentiment
_____________________________________
"""

def measure_finalSenti(row):
    """
    Positive(+1): when sentiment from both dictionaries are positive
    Negative(-1): when sentiment from both dictionaries are negative
    Neutral(0): Otherwise
    """
    if (row['Senti 1']*row['Senti 2'] >0):
        if row['Senti 1']>0:
            senti = 1
        else:
            senti = -1
    else:
        senti = 0
    
    return senti

data['Sentiment'] = data.apply(measure_finalSenti, axis=1) 
#axis=1 option sends the entire row to 'measure_finalsenti' function

data_final = data[['id','title','contents','Sentiment']]

data_final = data_final.to_json(orient='records',force_ascii=False)

with open('data/data_senti_dum.json', 'w') as f:
    f.write(data_final)
    

"""
_____________________________________________________________
Step 6: Measure the accuracy of sentiment based on word count
_____________________________________________________________
"""   

data_valid = pd.read_json('data/manualVerified_senti_2600.json')
data_wordcount = pd.read_json('data/data_senti_dum.json')

accuracy = (data_wordcount['Sentiment']==data_valid['Sentiment']).sum()/len(data_valid)
print("Accuracy of word count based sentiment = ", accuracy)

