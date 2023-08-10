import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
pd.set_option('display.max_colwidth', -1)
import re
from collections import Counter
import itertools

nlp = spacy.load('en')

# Data source is 290 downloaded articles from the Training Data
df = pd.read_csv('https://s3-us-west-1.amazonaws.com/simon.bedford/d4d/article_contents.csv')
df = df.fillna('')

# Specified reporting terms from challenge description
reporting_terms = [
    'displaced', 'evacuated', 'forced to flee', 'homeless', 'in relief camp',
    'sheltered', 'relocated', 'destroyed housing', 'partially destroyed housing',
    'uninhabitable housing'
]

# Specified reporting units from challenge description
reporting_units = {
    'people': ['people', 'persons', 'individuals', 'children', 'inhabitants', 'residents', 'migrants'],
    'households': ['families', 'households', 'houses', 'homes']
}

direct_phrases = []
nouns = 'people|persons|families|individuals|children|inhabitants|residents|migrants|villagers'
nouns = nouns.split("|")
verbs = 'evacuated|displaced|fled|forced to flee|relocated|forced to leave'
verbs = verbs.split("|")

for n, v in list(itertools.product(nouns, verbs)):
    direct_phrases.append(n + " " + v)
    direct_phrases.append(v + " " + n)

nouns = 'houses|homes'
nouns = nouns.split("|")
verbs = 'destroyed|damaged|flooded|inundated|lost|collapsed|submerged|washed away|affected|demolished'
verbs = verbs.split("|")

for n, v in list(itertools.product(nouns, verbs)):
    direct_phrases.append(n + " " + v)
    direct_phrases.append(v + " " + n)

#
housing_units = re.compile('houses|homes')
housing_impacts = re.compile("destroyed|damaged|flooded|inundated|lost|collapsed|submerged|washed away|affected|demolished")

#
people_units = re.compile('people|persons|families|individuals|children|inhabitants|residents|migrants')
people_impacts = re.compile('evacuated|displaced|fled|forced to flee|relocated|forced to leave')

units_regex = re.compile('households|houses|homes|people|persons|families|individuals|children|inhabitants|residents|migrants|villagers')
impacts_regex = re.compile('destroyed|damaged|flooded|inundated|lost|collapsed|submerged|washed away|affected|demolished|evacuated|displaced|fled|forced to flee|relocated|forced to leave')

def clean_string(s):
    return s.replace('\xa0', '')

# This is to test for phrases that are direct combinations of reporting units and terms, for example
# evaucated people / people evacuated
# destroyed houses / houses destroyed
# It is possible that some of these combinations could occur in irrelevant documents

def check_initial_combinations(article):
    article = article.lower()
    regex = re.compile("|".join(direct_phrases))
    if re.search(regex, article):
        return True

# Noun phrase with housing units and housing impacts, examples:
# at least 60 homes were destroyed across three districts
# mark gunning said 116 houses in wye river and separation creek had been destroyed
# the landslide, which covered about 2 sq km (0.8 sq miles), damaged at least 11 homes, xinhua reported.
# as more than 8,000 people were evacuated from their homes,
# some 2,500 people were evacuated from hard-hit grimma, near leipzig.

def get_noun_phrase_sentences(article, units_regex, impacts_regex):
    sentences = []
    doc = nlp(u"{}".format(article.lower()))
    for s in doc.sents:
        d = nlp(u"{}".format(s))
        for np in d.noun_chunks:
            if re.search(units_regex, np.text) and re.search(impacts_regex, np.root.head.text):
                sentences.append(str(s))
    return sentences

def check_noun_phrases(article, units_regex, impacts_regex):
    doc = nlp(u"{}".format(article.lower()))
    for s in doc.sents:
        d = nlp(u"{}".format(s))
        for np in d.noun_chunks:
            if re.search(units_regex, np.text) and re.search(impacts_regex, np.root.head.text):
                return True

# Combinations of relevant units and impacts, exmples:
# provide assistance to impacted and displaced families
# accommodation had been provided for about 600 displaced residents.
# it is expected that displaced families will need relief supplies

def get_simple_combinations_sentences(article, units_regex, impacts_regex):
    sentences = []
    doc = nlp(u"{}".format(article.lower()))
    for s in doc.sents:
        d = nlp(u"{}".format(s))
        for np in d.noun_chunks:
            if re.search(units_regex, np.text) and re.search(impacts_regex, np.text):
                sentences.append(str(s))
    return sentences

def check_simple_combinations(article, units_regex, impacts_regex):
    doc = nlp(u"{}".format(article.lower()))
    for s in doc.sents:
        d = nlp(u"{}".format(s))
        for np in d.noun_chunks:
            if re.search(units_regex, np.text) and re.search(impacts_regex, np.text):
                return True
            if re.search(units_regex, np.text) and re.search(impacts_regex, " ".join([l.text for l in np.rights])):
                return True

# Units as passive subjects, examples:
# Hundreds of homes have been destroyed in Algeria‘s southern city of Tamanrasset...
# confirming that 15 families have been evacuated to the town hall as a precaution against collapse.

def get_passive_subject_sentences(article, units_regex, impacts_regex):
    sentences = []
    doc = nlp(u"{}".format(article.lower()))
    for s in doc.sents:
        d = nlp(u"{}".format(s))
        for token in d:
            if re.search(impacts_regex, str(token)):
                children = [t for t in token.children]
                for c in children:
                    if c.dep_ in ('nsubjpass', 'nsubj'):
                        obj = " ".join([str(a) for a in c.subtree])
                        if re.search(units_regex, obj):
                            sentences.append(s)
    return sentences

def check_passive_subject(article, units_regex, impacts_regex):
    sentences = []
    doc = nlp(u"{}".format(article.lower()))
    for s in doc.sents:
        d = nlp(u"{}".format(s))
        for token in d:
            if re.search(impacts_regex, str(token)):
                children = [t for t in token.children]
                for c in children:
                    if c.dep_ in ('nsubjpass', 'nsubj'):
                        obj = " ".join([str(a) for a in c.subtree])
                        if re.search(units_regex, obj):
                            return True

# Examples
# displaced residents said they couldn't believe how quickly the situation escalated
# around 20,000 people had to be evacuated from their homes

def get_test1_sentences(article, units_regex, impacts_regex):
    sentences = []
    doc = nlp(u"{}".format(article.lower()))
    for s in doc.sents:
        d = nlp(u"{}".format(s))
        for token in d:
            if re.search(impacts_regex, str(token)):
                ancestors = [t for t in token.ancestors]
                for a in ancestors:
                    if a.dep_ == 'ROOT':
                        children = [c for c in a.children]
                        for c in children:
                            if c.dep_ == 'nsubj' and re.search(units_regex, str(c)):
                                sentences.append(s)
    return sentences

def test1(article, units_regex, impacts_regex):
    doc = nlp(u"{}".format(article.lower()))
    for s in doc.sents:
        d = nlp(u"{}".format(s))
        for token in d:
            if re.search(impacts_regex, str(token)):
                ancestors = [t for t in token.ancestors]
                for a in ancestors:
                    if a.dep_ == 'ROOT':
                        children = [c for c in a.children]
                        for c in children:
                            if c.dep_ == 'nsubj' and re.search(units_regex, str(c)):
                                return True

# Simple tests based upon combinations of words
# Unlikely to occur elsewhere
# Examples 'left homeless'
simple_phrases = [
    'left homeless',
    'families homeless',
    'people homeless',
    'residents homeless',
    'evacuate their homes',
    'left their homes',
    'fled to relief camps',
    'flee from their homes',
    'people evacuated',
    'houses damaged',
    'houses_submerged'
]
simple_phrases_regex = re.compile("|".join(simple_phrases))
def check_simple_phrases(article, simple_phrases_regex=simple_phrases_regex):
    if re.search(simple_phrases_regex, article.lower()):
        return True

def check_relevance(article):
    article = clean_string(article)
    if check_initial_combinations(article):
        return True
    if check_simple_phrases(article):
        return True
    if check_simple_combinations(article, units_regex, impacts_regex):
        return True
    if check_noun_phrases(article, units_regex, impacts_regex):
        return True
    if check_passive_subject(article, units_regex, impacts_regex):
        return True
    if test1(article, units_regex, impacts_regex):
        return True
    return False

df['is_relevant'] = df['content'].apply(check_relevance)

total = len(df)
relevant = (df['is_relevant'] == 1).sum()
print ("{:.0f}% identified as relevant".format(relevant/total * 100))

for i, row in (df[df['is_relevant'] == 0].head(10)).iterrows():
    print(row['content'])
    print("\n")



