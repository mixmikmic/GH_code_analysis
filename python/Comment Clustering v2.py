# Imports
from pickle import dump, load
import nltk
from nltk import word_tokenize,FreqDist
import re
from nltk.corpus import wordnet as wn
from nltk.util import ngrams
from sklearn.cluster import KMeans
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_distances
import pandas as pd
import json

doc_list =load(open("data/Master2_doc_content",'rb'))
len(doc_list)

# Start working on one document and associated comments
document = doc_list[0]
document.keys()

type(document['text'])

# Convert bs4 ResultSet to a list of strings
comments = []
for c in document['comment_list']:
    c = c.replace('\n',' ')
    comments.append(str(c))

len(comments)

# Modified from Brandon Rose:
def tokenize_text(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

def stem_text(text):
    tokens = tokenize_text(text)
    stems = [stemmer.stem(t) for t in tokens]
    return stems

stemmer = SnowballStemmer('english')

tfidf_vec = TfidfVectorizer(tokenizer=tokenize_text,
                            stop_words='english',
                            ngram_range=(1,3),
                            min_df=0.2, max_df=0.8,
                            max_features=200000)

tfidf_matrix = tfidf_vec.fit_transform(comments)

num_clusters = 12
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

cluster_center_list = []
for c in clusters:
    cluster_center_list.append(km.cluster_centers_[c])

center_distances = paired_distances(tfidf_matrix, cluster_center_list)

comment_clusters = {'comment': comments, 'cluster': clusters, 'dist': center_distances}
comment_frame = pd.DataFrame(comment_clusters, index = [clusters] , columns = ['comment', 'cluster', 'dist'])

comment_frame['cluster'].value_counts()

print(comment_frame[comment_frame.cluster==3].max())
print()
print(comment_frame[comment_frame.cluster==3].min())
print()
print(comment_frame[comment_frame.cluster==5].min())
print()
list(comment_frame[comment_frame.cluster==3]['comment'])

print('Most Central Comments by Cluster\n')
for i in range(num_clusters):
    print('Cluster {}\n'.format(i))
    print(comment_frame[comment_frame.cluster==i].min().comment)
    print()

# Modified from Brandon Rose and
# http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html
def vocabulary_frame(text):
    tokens = tokenize_text(text)
    stems = stem_text(text)
    return pd.DataFrame({'words': tokens}, index = stems).drop_duplicates()

def extended_vocabulary_frame(texts):
    frames = []
    for t in texts:
        vf = vocabulary_frame(t)
        frames.append(vf)
    extended = pd.concat(frames).drop_duplicates()
    return extended

def km_print_top_words(model, num_clusters, vocab_frame, feature_names, n_top_words):    
    print("Top terms per cluster:\n")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1] 
    for i in range(num_clusters):
        print("Cluster %d Words:" % i, end=' ')
        top_words = []
        top_words.append(vocab_frame.ix[feature_names[ind].split(' ')].values.tolist()[0][0] for ind in order_centroids[i, :n_top_words])
        print(top_words)
        print()

def process_document(document):
    comments = []
    for c in document['comment_list']:
        c = c.replace('\n',' ')
        comments.append(str(c))
    return comments

# Modified from Brandon Rose:
def tokenize_text(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

def stem_text(text):
    stemmer = SnowballStemmer('english')
    tokens = tokenize_text(text)
    stems = [stemmer.stem(t) for t in tokens]
    return stems

def vectorize_comments(comments):
    tfidf_vec = TfidfVectorizer(tokenizer=tokenize_text,
                                stop_words='english',
                                ngram_range=(1,3),
                                min_df=0.2, max_df=0.8,
                                max_features=200000)
    tfidf_matrix = tfidf_vec.fit_transform(comments)
    return tfidf_matrix, tfidf_vec

# Modified from Brandon Rose:
def vocabulary_frame(text):
    tokens = tokenize_text(text)
    stems = stem_text(text)
    return pd.DataFrame({'words': tokens}, index = stems).drop_duplicates()

def extended_vocabulary_frame(texts):
    frames = []
    for t in texts:
        vf = vocabulary_frame(t)
        frames.append(vf)
    extended = pd.concat(frames).drop_duplicates()
    return extended

def top_words(model, num_clusters, comments, tfidf_vec, n_top_words):
    feature_names = tfidf_vec.get_feature_names()
    comment_vf = extended_vocabulary_frame(comments)
    order_centroids = model.cluster_centers_.argsort()[:, ::-1] 
    top_words = []
    for i in range(num_clusters):
        temp_top_words = []
        temp_top_words.append(vocab_frame.ix[feature_names[ind].split(' ')].values.tolist()[0][0]
                              for ind in order_centroids[i, :n_top_words])
        top_words.append(temp_top_words)



def cluster_comments(document, num_clusters):
    cluster_dict = {}

    comments = process_document(document)
    tfidf_matrix, tfidf_vec = vectorize_comments(comments)
    
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    
    cluster_center_list = []
    for c in clusters:
        cluster_center_list.append(km.cluster_centers_[c])
    center_distances = paired_distances(tfidf_matrix, cluster_center_list)
    
    comment_clusters = {'comment': comments, 'cluster': clusters, 'dist': center_distances}
    comment_frame = pd.DataFrame(comment_clusters, index = [clusters] , columns = ['comment', 'cluster', 'dist'])
    
    central_comments = []
    all_comments = []
    for i in range(num_clusters):
        central_comments.append(comment_frame[comment_frame.cluster==i].min().comment)
        all_comments.append(list(comment_frame[comment_frame.cluster==i]['comment']))
    
    freq_words = top_words(km, num_clusters, comments, tfidf_vec, 6)
    
    cluster_dict['central_comments'] = central_comments
    cluster_dict['all_comments'] = all_comments
    cluster_dict['top_words'] = freq_words
    
    return cluster_dict

test = cluster_comments(document, 12)

test.keys()

type(test['top_words'])

feature_names = tfidf_vec.get_feature_names()
comment_vf = extended_vocabulary_frame(comments)
km_print_top_words(km, num_clusters, comment_vf, feature_names, 6)

json_data = []

doc_list1 =load(open("data/Master_doc_content",'rb'))
doc_list2 = load(open("data/Master2_doc_content",'rb'))

doc_id1 = ["FAA-2010-1127-0001","USCBP-2007-0064-1986","FMCSA-2015-0419-0001","NARA-06-0007-0001","APHIS-2006-0041-0001","EBSA-2012-0031-0001","IRS-2010-0009-0001","BOR-2008-0004-0001","OSHA-2013-0023-1443","DOL-2016-0001-0001","NRC-2015-0057-0086","CMS-2010-0259-0001","CMS-2009-0008-0003","CMS-2009-0038-0002","NPS-2014-0005-000","BIS-2015-0011-0001","HUD-2011-0056-0019","HUD-2011-0014-0001","OCC-2011-0002-0001","ACF-2015-0008-0124","ETA-2008-0003-0001","CMS-2012-0152-0004","CFPB-2013-0033-0001","USCIS-2016-0001-0001","FMCSA-2011-0146-0001","USCG-2013-0915-0001","NHTSA-2012-0177-0001","USCBP-2005-0005-0001"]
doc_id2 = ["HUD-2015-0101-0001","ACF-2010-0003-0001","NPS-2015-0008-0001","FAR-2014-0025-0026","CFPB-2013-0002-0001","DOS-2010-0035-0001","USCG-2013-0915-0001","SBA-2010-0001-0001"]

doc_title1 = ["Photo Requirements for Pilot Certificates",
             "Advance Information on Private Aircraft Arriving and Departing the United States",
             "Evaluation of Safety Sensitive Personnel for Moderate-to-Severe Obstructive Sleep Apnea",
             "Changes in NARA Research Room and Museum Hours",
             "Bovine Spongiform Encephalopathy; Minimal-Risk Regions; Importation of Live Bovines and Products Derived From Bovines",
             "Incentives for Nondiscriminatory Wellness Programs in Group Health Plans",
             "Furnishing Identifying Number of Tax Return Preparer",
             "Use of Bureau of Reclamation Land, Facilities, and Waterbodies",
             "Improve Tracking of Workplace Injuries and Illnesses",
             "Implementation of the Nondiscrimination and Equal Opportunity Provisions of the Workforce Innovation and Opportunity Act",
             "Linear No-Threshold Model and Standards for Protection Against Radiation; Extension of Comment Period",
             "Medicare Program: Accountable Care Organizations and the Medicare Shared Saving Program",
             "Medicare Program: Changes to the Competitive Acquisition of Certain Durable Medical Equipment, Prosthetics, Orthotics and Supplies (DMEPOS) by Certain Provisions of the Medicare Improvements for Patients and Providers Act of 2008 (MIPPA)",
             "Medicare Program: Inpatient Rehabilitation Facility Prospective Payment System for Federal Fiscal Year 2010 ",
             "Special Regulations: Areas of the National Park System, Cuyahoga Valley National Park, Bicycling",
             "Wassenaar Arrangement Plenary Agreements Implementation; Intrusion and Surveillance Items",
             "Credit Risk Retention 2",
             "FR 5359–P–01 Equal Access to Housing in HUD Programs Regardless of Sexual Orientation or Gender Identity ",
             "Credit Risk Retention",
             "Head Start Performance Standards; Extension of Comment Period",
             "Senior Community Service Employment Program",
             "Patient Protection and Affordable Care Act: Benefit and Payment Parameters for 2014",
             "Debt Collection (Regulation F)",
             "U.S. Citizenship and Immigration Services Fee Schedule",
             "Applicability of Regulations to Operators of Certain Farm Vehicles and Off-Road Agricultural Equipment",
             "Carriage of Conditionally Permitted Shale Gas Extraction Waste Water in Bulk",
             "Federal Motor Vehicle Safety Standards: Event Data Recorders",
             "Documents Required for Travel Within the Western Hemisphere"]
doc_title2 = ["FR 5597-P-02 Instituting Smoke- Free Public Housing",
             "Head Start Program",
             "Off-Road Vehicle Management: Cape Lookout National Seashore",
             "Federal Acquisition Regulations: Fair Pay and Safe Workplaces; Second Extension of Time for Comments (FAR Case 2014-025)",
             "Ability to Repay Standards under Truth in Lending Act (Regulation Z)",
             "Schedule of Fees for Consular Services, Department of State and Overseas Embassies and Consulates",
             "Carriage of Conditionally Permitted Shale Gas Extraction Waste Water in Bulk",
             "Women-Owned Small Business Federal Contract Program"]

for i in range(len(doc_list1)):
    print(i)
    info_dic = {}
    doc_text = str(doc_list1[i]['text'][0])
    info_dic["keywords"],info_dic["sentences"], info_dic["summary"] = extract_summary(doc_text)
    info_dic["doc_id"], info_dic["doc_title"] = doc_id1[i], doc_title1[i]
    data.append(info_dic)

for i in range(len(doc_list2)):
    print(i)
    info_dic = {}
    doc_text = str(doc_list2[i]['text'][0])
    info_dic["keywords"],info_dic["sentences"], info_dic["summary"] = extract_summary(doc_text)
    info_dic["doc_id"], info_dic["doc_title"] = doc_id2[i], doc_title2[i]
    data.append(info_dic)

top_obj = {}
top_obj['data'] = json_data

with open('data/comment_data.json', 'w') as outfile:
    json.dump(top_obj, outfile)

