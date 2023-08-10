from IPython.display import Image, display
display(Image(filename='img/Preview of Osha.png', embed=True))

display(Image(filename='img/Preview of Msia.png', embed=True))

display(Image(filename='img/Combination of Main Causes.png', embed=True))

display(Image(filename='img/Figure 4 - Distribution of Causes.png', embed=True))

display(Image(filename='img/Figure 5 - Distribution of Causes after Data Supplementation.png', embed=True))

import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.tag.stanford import StanfordPOSTagger

# load data file
msia = pd.read_excel('data/MsiaAccidentCases.xlsx')
osha = pd.read_excel('data/osha.xlsx', header=None, parse_cols='A:D')

msia.head(3)

osha.head(3)

osha[osha[1]==" Electric Shock "].head(3)

osha[osha[2]=="InspectionOpen DateSICEstablishment Name"].head(3)

# standardized dataframe header
msia.columns = [name.split()[0] for name in msia.columns]
osha.columns = ['Accident', 'Title', 'Summary', 'Keywords']

# merge Other and Others by changing Others to Other
msia.loc[msia.Cause == 'Others', 'Cause'] = 'Other'

# Strip whitespace (including newlines) from each string in the Series/Index from left and right sides
for col in ['Title', 'Summary', 'Keywords']:
    osha[col] = osha[col].astype('str').str.strip()

# impute missing data from source website by recrawling
url = 'https://www.osha.gov/pls/imis/accidentsearch.accident_detail'
# find index containing missing value
is_missing = (osha.Summary == 'InspectionOpen DateSICEstablishment Name') | (
    osha.Title.apply(lambda x: x.count(' ') <= 1))
length = sum(is_missing)
count = 0
print('Recrawling data for imputing...')
for index, row in osha.loc[is_missing].iterrows():
    print('%d/%d' % (count, length))
    page = requests.get(url, params={'id': row.Accident})  # get source page of missing record

    soup = BeautifulSoup(page.content, 'html.parser')
    strong = soup.select('#maincontain > div > p.text-center > strong')[0]
    td = soup.findAll("td", {"colspan": "8"})

    title = strong.text[strong.text.index('-') + 1:].strip()
    summary = td[1].text.strip()
    keywords = td[2].text[9:].strip().split(', ')

    osha.loc[index, 'Title'] = title
    osha.loc[index, 'Summary'] = summary
    osha.loc[index, 'Keywords'] = '  '.join(keywords)  # double spaces
    count += 1

# Using Stanford POS Tagger rather than nltk default pos tagger to get more accurate pos tag
pos_model_path = 'C:/tools/stanford-postagger-full-2017-06-09/models/english-bidirectional-distsim.tagger'  # change folder path to your own
pos_jar_path = 'C:/tools/stanford-postagger-full-2017-06-09/stanford-postagger.jar'  # change folder path to your own
# Initialize the tagger
st_pos = StanfordPOSTagger(pos_model_path, pos_jar_path)

print('Start Standford POS Tagging for Title...')
osha['Title_POS'] = osha.Title.apply(lambda x: st_pos.tag(word_tokenize(x.lower())))
osha.to_csv(output_dir + 'osha_cleaned_pos.csv', index=False)
print('Start Standford POS Tagging for Summary...')
osha['Summary_POS'] = osha.Summary.apply(lambda x: st_pos.tag(word_tokenize(x.lower())))
osha.to_csv(output_dir + 'osha_cleaned_pos.csv', index=False)

# save to file
msia.to_csv(output_dir + 'MsiaAccidentCases_cleaned.csv', index=False)

msia = pd.read_csv( 'data/MsiaAccidentCases_cleaned.csv')
osha = pd.read_csv( 'data/osha_cleaned_pos.csv')

import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

wnl = nltk.WordNetLemmatizer()


def get_wordnet_pos(treebank_tag):
    '''
    Map the treebank tags to WordNet part of speech names.
    :param treebank_tag: nltk default pos tag
    :return: wordnet pos tag
    '''
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # including startswith('N') and unknown pos


def lemmatize(text):
    '''
    Lemmatizing.
    :param text: string
    :return: lemmatized string
    '''
    text = text.lower()

    # TODO check if need sentence tokenization
    pos_tokens = [(token, pos) for token, pos in pos_tag(word_tokenize(text)) if
                  token.isalpha() and token not in stopwords.words('english')]
    lem_text = ' '.join([wnl.lemmatize(token, get_wordnet_pos(pos)) for token, pos in pos_tokens])

    return lem_text

print('Training models based on Msia Accident Cases Title...')
title_final = msia['Title'].apply(lemmatize)

# separate dataset
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(title_final)
y = msia.Cause

X_train = X[:222]
X_test = X[222:]
y_train = y[:222]
y_test = y[222:]

seed = 35
dt = DecisionTreeClassifier(random_state=seed).fit(X_train, y_train)
print('\tDecision Tree:\t\t\t%f' % dt.score(X_test, y_test))

knn = KNeighborsClassifier(n_neighbors=11, weights='distance',                            metric='cosine', algorithm='brute').fit(X_train, y_train)
print('\tK Nearest Neighbors:\t\t%f' % knn.score(X_test, y_test))

mnb = MultinomialNB().fit(X_train, y_train)
print('\tNaive Bayesian:\t\t\t%f' % mnb.score(X_test, y_test))

svm = SVC(C=1000000.0, gamma='auto', kernel='rbf').fit(X_train, y_train)
print('\tSVM:\t\t\t\t%f' % svm.score(X_test, y_test))

lr = LogisticRegression().fit(X_train, y_train)
print('\tLogistic Regression:\t\t%f' % lr.score(X_test, y_test))

vc = VotingClassifier(estimators=[     ('dt', dt), ('knn', knn), ('mnb', mnb), ('svm', svm), ('lr', lr)     ], voting='hard').fit(X_train, y_train)

print('\tEnsemble (Majority Vote):\t%f' % vc.score(X_test, y_test))

print('Prediction score based on Summary:')
lem_summary = msia['Summary'].apply(lemmatize)
vectorizer2 = TfidfVectorizer()
X2 = vectorizer2.fit_transform(lem_summary)
y2 = msia.Cause

X2_train = X2[:222]
X2_test = X2[222:]
y2_train = y2[:222]
y2_test = y2[222:]

dt2 = DecisionTreeClassifier(random_state=seed).fit(X2_train, y2_train)
print('\tDecision Tree:\t\t\t%f' % dt2.score(X2_test, y2_test))

knn2 = KNeighborsClassifier(n_neighbors=11, weights='distance',                             metric='cosine', algorithm='brute').fit(X2_train, y2_train)
print('\tK Nearest Neighbors:\t\t%f' % knn2.score(X2_test, y2_test))

mnb2 = MultinomialNB().fit(X2_train, y2_train)
print('\tNaive Bayesian:\t\t\t%f' % mnb2.score(X2_test, y2_test))

svm2 = SVC(C=1000000.0, gamma='auto', kernel='rbf').fit(X2_train, y2_train)
print('\tSVM:\t\t\t\t%f' % svm2.score(X2_test, y2_test))

lr2 = LogisticRegression().fit(X2_train, y2_train)
print('\tLogistic Regression:\t\t%f' % lr2.score(X2_test, y2_test))

vc2 = VotingClassifier(estimators=[     ('dt', dt2), ('knn', knn2), ('mnb', mnb2), ('svm', svm2), ('lr', lr2)     ], voting='hard').fit(X2_train, y2_train)

print('\tEnsemble (Majority Vote):\t%f' % vc2.score(X2_test, y2_test))
print

print('Using SVM Model based on Titles of Msia dataset to predice Causes for OSHA dataset...')
print

# predict
lem_title_osha = osha['Title'].apply(lemmatize)
X_osha = vectorizer.transform(lem_title_osha)
osha_pred = svm.predict(X_osha)
osha['Cause'] = pd.Series(osha_pred)

print('Distribution of causes for OSHA Accident Cases dataset (predicted):')
osha_cause_count = osha.groupby('Cause').size().sort_values(ascending=False)
osha_cause_count.plot(kind='barh')
plt.show()
print(osha_cause_count)

# osha.to_csv('data/osha_cleaned_pos_predict.csv', index=False)

display(Image(filename='img/Figure 7 - Overview of Processa Processing.png', embed=True))

import pandas as pd
import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud
from ast import literal_eval

wnl = nltk.WordNetLemmatizer()  # initialize Lemmatizer

# chunk sequences of proper nouns phrase
grammar_title = '''
NP: {<NN.*>+}
CLAUSE: {^<CD>?<NP><POS>?<NN.*>*<VB.*>}
'''
cp_title = nltk.RegexpParser(grammar_title)

grammar_summary = '''
NP: {<NN.*>+}
CLAUSE: {<\#><CD><DT>?<JJ>*<NP>}
'''
cp_summary = nltk.RegexpParser(grammar_summary)

stops = ['employee', 'worker', 'laborer', 'owner', 'coworker', 'contractor']  # meaningless occupation term list
errors = ['age', 'hand', 'male']


def trasverse_tree(chunked):
    for n1 in chunked:
        if isinstance(n1, nltk.tree.Tree) and n1.label() == 'CLAUSE':
            for n2 in n1:
                if isinstance(n2, nltk.tree.Tree) and n2.label() == 'NP':
                    lem = [wnl.lemmatize(w) for w, t in n2]
                    # reject 'worker' but accept 'farm worker'
                    if lem[0] not in (stops + errors):
                        if len(lem) > 1 and lem[-1] not in (stops + errors):
                            lem = [lem[-1]]
                        return ' '.join(lem)


def parse_title_occupation(text):
    '''
    Extract occupation information from input
    :param text: string storing pos tag
    :return: occupation term
    '''
    pos = literal_eval(text)  # string to list
    chunked = cp_title.parse(pos)  # chunking

    return trasverse_tree(chunked)


def parse_summary_occupation(text):
    pos = literal_eval(text)  # string to list
    try:
        first_sent = pos[:pos.index(('.', '.'))]  # parse first sentence with pos tag
    except ValueError:
        return None
    chunked = cp_summary.parse(first_sent)  # chunking

    return trasverse_tree(chunked)


print('Parsing Title ...')
osha_title_occupation = osha.Title_POS.apply(parse_title_occupation)

print('Parsing Summary ...')
osha_summary_occupation = osha[osha_title_occupation.isnull()]['Summary_POS'].apply(parse_summary_occupation)

osha_title_occupation = osha_title_occupation.dropna()  # remove all None value
osha_summary_occupation = osha_summary_occupation.dropna()  # remove all None value
osha_occupation = pd.concat([osha_title_occupation, osha_summary_occupation])  # Concatenating together

# visualization
print('Top 10 risky occupations for OSHA dataset:')
osha_occupation_count = osha_occupation.groupby(osha_occupation).size().sort_values(ascending=False)
osha_occupation_count.head(10).plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()
print(osha_occupation_count.head(20))
print('OSHA occupations word cloud:')
osha_word_string = ' '.join([w.replace(' ', '_') for w in osha_occupation])
osha_word_cloud = WordCloud().generate(osha_word_string)
plt.imshow(osha_word_cloud)
plt.axis('off')
plt.show()

display(Image(filename='img/Figure 10 - Overview of Data Processing.png', embed=True))

import re
import pickle
from collections import Counter
import numpy as np

# load body terms list
try:
    human_body_terms = pickle.load(open("data/human_body_terms.pk", "rb"))
except FileNotFoundError:
    page = requests.get('http://www.enchantedlearning.com/wordlist/body.shtml')
    soup = BeautifulSoup(page.content, 'html.parser')
    table = soup.find('table', {'border': '1'})
    tds = table.findAll("td")
    [center.extract() for td in tds for center in td.findAll('center')]
    human_body_terms = [term for td in tds for term in re.split('\n+', td.text.strip())]
    pickle.dump(human_body_terms, open("data/human_body_terms.pk", "wb"))


def get_body_parts(text):
    '''
    Extract human body term from input
    :param text: str
    :return: list containing body terms
    '''
    keywords = text.split()
    body_parts = [k for k in keywords if k in human_body_terms]
    if body_parts:
        return body_parts


if __name__ == "__main__":
    osha_body_parts = osha.Keywords.apply(get_body_parts)

    osha_body_parts = osha_body_parts.dropna()  # remove all None value

    # visualization
    print('Top 10 common injured of human body parts for OSHA dataset:')
    osha_body_parts_statistic = {}
    for parts in osha_body_parts:
        for part in parts:
            if part not in osha_body_parts_statistic:
                osha_body_parts_statistic.update({part: 1})
            else:
                osha_body_parts_statistic[part] += 1
    top_10 = Counter(osha_body_parts_statistic).most_common(10)  # find top 10 common injured of human body parts
    objects = tuple([o for o, c in top_10])
    y_pos = np.arange(len(objects))
    count = [c for o, c in top_10]
    plt.bar(y_pos, count)
    plt.xticks(y_pos, objects)
    plt.ylabel('Amount')
    plt.title('Top 10 common injured of human body parts')
    plt.show()
    for o, c in top_10: print('%s:\t%d' % (o, c))

    print('OSHA injured of human body parts word cloud:')
    osha_word_string = ' '.join([' '.join(w) for w in osha_body_parts])
    
    osha_word_cloud = WordCloud().generate_from_frequencies(osha_body_parts_statistic)
    plt.imshow(osha_word_cloud)
    plt.axis('off')
    plt.show()
    print

