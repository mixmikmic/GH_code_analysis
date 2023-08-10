import os, re, string
from bs4 import BeautifulSoup
from bs4.element import Comment

# spacy is used for Part of Speech tagging and Named Entity Recognition
# spacy is a non-standard python library which can be installed using 'pip install spacy' from the command line
# language models can be downloaded by running 'python -m spacy download <language>' from the command line
import spacy
language = 'en'
nlp_model  = spacy.load('en')
    
def get_entities(text):
    
    doc = nlp_model(text)
    labels = [{ent.text:ent.label_} for ent in doc.ents]

    return labels

path_to_docs = '../SPLC_Scrape_Results/'
output_clean = 'clean_scraped_text/clean_scraped_text.txt'

def stripTags(text):
    scripts = re.compile(r'<script.*?/script>')
    css = re.compile(r'<style.*?/style>')
    tags = re.compile(r'<.*?>')

    text = scripts.sub('', text)
    text = css.sub('', text)
    text = tags.sub('', text)

    return text

def punctuation_remove(text):
    """
    Mutates and returns text where all punctuation are replaced
    """
    chars = re.escape(string.punctuation)
    return re.sub(r'['+chars+']', ' ',text)

def doublespace_remove(text):
    return re.sub(' +',' ',text)

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)

webfiles = [webfile for webfile in os.listdir(path_to_docs) if '.' in webfile]
htmlfiles = [{webfile:htmlfile} for htmlfile in os.listdir(path_to_docs + webfile) if htmlfile.endswith('.html') for webfile in webfiles]

ignore = ['DataDive/Google_Search_Masterlist_Identifier',
'PageRank',
'Reddit',
'preDive',
'Chan4_Analysis.R',
'LSmithscrape_paths_from_home_pages.R',
'README.md',
'SPLC_Scrape_and_Strip.ipynb',
'posts_4chan_pol_2.csv',
'posts_4chan_pol_delim.csv',
'.gitignore','.DS_Store', '.git', '.ipynb_checkpoints','RawUrls.txt']

characters_to_replace = ['\u']
for webfile in webfiles:
    if not webfile in ignore:
        htmldict = {}
        htmlfiles = [htmlfile for htmlfile in os.listdir(path_to_docs + webfile) if htmlfile.endswith('.html')]
        for htmlfile in htmlfiles:
            htmldict[webfile] = {}
            htmldict[webfile][htmlfile] = {}
            with open(path_to_docs + webfile + '/' + htmlfile, "r") as myfile:
                result = myfile.read()
            htmldict[webfile][htmlfile]['text'] = text_from_html(result)
            entities = get_entities(htmldict[webfile][htmlfile]['text'])
            for char in characters_to_replace:
                htmldict[webfile][htmlfile]['text'] = htmldict[webfile][htmlfile]['text'].encode('ascii','replace').lower().replace(char," ")
            htmldict[webfile][htmlfile]['text'] = punctuation_remove(htmldict[webfile][htmlfile]['text'])
            htmldict[webfile][htmlfile]['text'] = doublespace_remove(htmldict[webfile][htmlfile]['text'])
            htmldict[webfile][htmlfile]['entities'] = entities
            with open(output_clean, "a") as myfile:
                myfile.write(str(htmldict))
                myfile.write('\n')

htmldict['americanfreedomunion.com']['index.html']['entities'][0:10]

htmldict['americanfreedomunion.com']['index.html']['text'][0:2000]



