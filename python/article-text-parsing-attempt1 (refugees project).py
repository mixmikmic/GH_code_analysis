import pandas as pd

import spacy
nlp = spacy.load('en')

# the csv file is from the training data
articleDF = pd.read_csv('datasets/article_contents.csv')

articleDF.head(5)

print(articleDF['title'][0])

print(articleDF['meta_description'][0])

articleDF['content'][0]

articleDF['tag'].unique()

parsed_text = nlp(articleDF['content'][0])
#after running through spacy, some stuff you can do e.g. POS-- 
"""for word in parsed_text:
    print(word.pos_, word)"""

#trying to get all the numbers in the article:
#not too bad but need some sort of thinking as to deciding which ones are releavnt!
ents = list(parsed_text.ents)
for entity in ents:
    sentence = entity.sent
    for idx, word in enumerate(sentence):
        if(word == entity.root and entity.label_ == 'CARDINAL'):
            print(entity,sentence[idx + 1: idx + 3])

'''
Ignore this for now. I stumbled across a tutuorial for a text processing pipeline using the below-- might come in 
handy later... Except I can't seem to find the link...!
'''
def punct_space(token):
    """
    helper function to eliminate tokens
    that are pure punctuation or whitespace
    """
    
    return token.is_punct or token.is_space

def article_review(filename):
    """
    generator function to read in articles from the dataframe
    and un-escape the original line breaks in the text
    """
    
    with codecs.open(filename, encoding='utf_8') as f:
        for review in f:
            yield review.replace('\\n', '\n')
            
def lemmatized_sentence_corpus(filename):
    """
    generator function to use spaCy to parse reviews,
    lemmatize the text, and yield sentences
    """
    
    for parsed_article in nlp.pipe(article_review(filename), batch_size=10000, n_threads=4):
        
        for sent in parsed_article.sents:
            yield u' '.join([token.lemma_ for token in sent
                             if not punct_space(token)])

# intuition is that the main info would be in the title-- for now just parse 
# this rather than trying to deal with the whole aritlce
def parse_title_for_units(title_text):
    parsed_text = nlp(title_text) #run title through spacy pipline
    ents = list(parsed_text.ents) #grab the named entities detected
    output = []
    for entity in ents:
         #the sentence in whcih the entity lives-- since for numbers, spacy doesn't include units e.g. 160 rather than 160 peopel
        sentence = entity.sent
        for idx, word in enumerate(sentence): 
            # for now, it seems that the only the ones with entity label == cardinal are relevant to the number of reporting units
            if(word == entity.root and entity.label_ == 'CARDINAL'):
                # the following to grab a few extra words round that entity to get the units
                if(len(sentence) > idx + 1):
                    output.append(entity.text_with_ws + sentence[idx + 1].text)
                else:
                    output.append(sentence[idx - 1].text_with_ws + entity.text)
    return output

    

#test the parser for the first title...
parse_title_for_units(articleDF['title'][0])

#seems okay-- run it through the dataframe
articleDF['parse_title_for_units'] = articleDF['title'].map(parse_title_for_units)

articleDF.head(15)



articleDF.to_csv('datasets/refugess_training_text_parsing_test_1.csv')

articleDF[40:50]



