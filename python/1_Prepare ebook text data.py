"""
Example sentences to test spaCy and its language models.
>>> from spacy.lang.sv.examples import sentences
>>> docs = nlp.pipe(sentences)
"""

import spacy

from nltk.tokenize import sent_tokenize

#from spacy.lang.sv import Swedish
#nlp = Swedish()  # use directly
#nlp = spacy.blank('fi')  # blank instance

# English
# nlp = spacy.load('en')

with open('books/18043-0.txt', 'r') as f: 
    data = f.read()

import re
#clean_data = re.sub(r'[^a-zA-Z0-9-_*.,?!åäöèÅÄÖÈÉ]', ' ', data)
paragraphs = data.split("\n\n")
len(paragraphs)

paragraphs

# doc = nlp(data)

paragraph_sentence_list = []
for paragraph in paragraphs:
    paragraph = paragraph.replace("\n", " ")
    paragraph = paragraph.replace("--", "")
    paragraph = re.sub(r'[^a-zA-Z0-9_*.,?!åäöèÅÄÖÈÉçëË]', ' ', paragraph)
    paragraph_sentence_list.append(sent_tokenize(paragraph))

len(paragraph_sentence_list)

paragraph_sentence_list

text = ""
count = 0
for paragraph in paragraph_sentence_list:
    if " ".join(paragraph).isupper():
        with open("books/18043-0_aeneas_data_"+str(count)+".txt", "w") as fw:
            fw.write(text)
        text = ""
        count += 1
        text += "\n".join(paragraph)
        text += "\n\n"
    elif "End of the Project Gutenberg EBook" in " ".join(paragraph):
        break
    else:
        text += "\n".join(paragraph)
        text += "\n\n"    





