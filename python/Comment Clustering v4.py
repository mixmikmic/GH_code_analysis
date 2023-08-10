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

#cluster count
def get_document_text(raw_text):
    """ This function takes in raw document text as input which we receive from the API and returns a clean text 
    of the associated document. It cleans up any HTML code in the text, newline characters, and extracts supplemental
    information part of the document.
    
    INPUT: string
    OUTPUT: string
    """
    raw_text = raw_text.replace('\n',' ')
    raw_text = raw_text.replace('*','') # added
    raw_text = raw_text.replace('\r',' ') # added
    # Remove any residual HTML tags in text
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_text)
    return cleantext

def tokenize_text(corpus):
    pattern = r'''(?x)     # set flag to allow verbose regexps
    ((?:[A-Z]\.)+)         # abbreviations, e.g. B.C.
    | (?:(\w+([-']\w+))+)  # words with optional internal hyphens e.g. after-ages or author's
    | ([a-zA-Z]+)          # capture everything else
    '''
    tokens = nltk.regexp_tokenize(corpus,pattern)
    all_token = [word.lower() for token in tokens for word in token if word != "" and word[0] != "'" and word[0] != "-"]
    return all_token

def tokenize_text_sent(corpus):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sents = sent_tokenizer.tokenize(corpus) # Split text into sentences    
    return [tokenize_text(sent) for sent in raw_sents]

def tag_my_text(sents):
    return [nltk.pos_tag(sent) for sent in sents]

#Chunk noun phrases in tree 
def noun_phrase_chunker():
    grammar = r"""
    NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
    """
    cp = nltk.RegexpParser(grammar)
    return cp

#Extract only the NP marked phrases from the parse tree, that is the chunk we defined
def noun_phrase_extractor(sentences, chunker):
    res = []
    for sent in sentences:
        tree = chunker.parse(sent)
        for subtree in tree.subtrees():
            if subtree.label() == 'NP' : 
                res.append(subtree[0:len(subtree)])
                #res.append(subtree[0])
                #print(subtree)
    return res

#remove tags and get only the noun phrases , can be adjusted for length
def noun_phrase_finder(tagged_text):
    all_proper_noun = noun_phrase_extractor(tagged_text,noun_phrase_chunker()) 
    #does not literally mean proper noun. Chunker only extracts common noun
    noun_phrase_list = []                                                      
    #noun_phrase_string_list =[]
    for noun_phrase in all_proper_noun:
        if len(noun_phrase) > 0: #this means where the size of the phrase is greater than 1
            small_list =[]
            for (word,tag) in noun_phrase:
                small_list.append(word)
            noun_phrase_list.append(small_list)
            #noun_phrase_string_list.append(' '.join(small_list))
    return noun_phrase_list

#get freq dist obj for noun phrase of different lengths
def find_freq(nested_list,nest_len):
    #from nltk.probability import FreqDist
    fdist_list =[]
    for inner_np in nested_list:
        if len(inner_np) == nest_len:
            fdist_list.append(' '.join(inner_np))
    fdist = FreqDist(fdist_list)
    return fdist

def get_top_unigrams(np):
    unigrams = []
    for item in np:
        if len(item) ==  1:
            unigrams.append(item)
    fdist_uni = find_freq(np,1)
    uni_list = fdist_uni.most_common()
    threshold = 0.3 * len(unigrams)
    top = []
    s = 0
    for word,count in uni_list:
        top.append(word)
        s += count
        if s > threshold:
            break      
    return top

# Lesk algorithm for disambiguation in case of multiple synsets of a word
def compare_overlaps_greedy(context, synsets_signatures, pos=None):
    """
    Calculate overlaps between the context sentence and the synset_signature
    and returns the synset with the highest overlap.
    
    :param context: ``context_sentence`` The context sentence where the ambiguous word occurs.
    :param synsets_signatures: ``dictionary`` A list of words that 'signifies' the ambiguous word.
    :param pos: ``pos`` A specified Part-of-Speech (POS).
    :return: ``lesk_sense`` The Synset() object with the highest signature overlaps.
    """
    # if this returns none that means that there is no overlap
    max_overlaps = 0
    lesk_sense = None
    for ss in synsets_signatures:
        if pos and str(ss.pos()) != pos: # Skips different POS.
            continue
        overlaps = set(synsets_signatures[ss]).intersection(context)
        if len(overlaps) > max_overlaps:
            lesk_sense = ss
            max_overlaps = len(overlaps)  
    return lesk_sense

def lesk(context_sentence, ambiguous_word, pos=None, dictionary=None):
    """
    This function is the implementation of the original Lesk algorithm (1986).
    It requires a dictionary which contains the definition of the different
    sense of each word. See http://goo.gl/8TB15w

        >>> from nltk import word_tokenize
        >>> sent = word_tokenize("I went to the bank to deposit money.")
        >>> word = "bank"
        >>> pos = "n"
        >>> lesk(sent, word, pos)
        Synset('bank.n.07')
    
    :param context_sentence: The context sentence where the ambiguous word occurs.
    :param ambiguous_word: The ambiguous word that requires WSD.
    :param pos: A specified Part-of-Speech (POS).
    :param dictionary: A list of words that 'signifies' the ambiguous word.
    :return: ``lesk_sense`` The Synset() object with the highest signature overlaps.
    """
    if not dictionary:
        dictionary = {}
        for ss in wn.synsets(ambiguous_word):
            dictionary[ss] = ss.definition().split()
    best_sense = compare_overlaps_greedy(context_sentence, dictionary, pos)
    return best_sense
    #return dictionary 

# this function takes in a word and gets the most relevant synset based on context from the text. 
# for exact algorithm, refer the text above ("what I want to do" markdown)
def get_synset(word,pos_tag_text ,pos):
    if len(wn.synsets(word)) == 1:
        #print("here1")
        return wn.synsets(word)[0]
    else:
        #get all context sentences
        all_sent =[]
        for sent in pos_tag_text:
            for (w,t) in sent:
                if w == word:
                    all_sent.append(sent)
        #call lesk here
        app_syn = lesk(all_sent[len(all_sent)//2], word, pos)
        if app_syn != None:
            #print("here2")
            return app_syn
        else:
            #second lesk trial with another context sentence
            app_syn = lesk(all_sent[len(all_sent)//3], word, pos)
            if app_syn != None:
                #print("here2")
                return app_syn
            else:
                #give up and choose 1st synset from list with matching pos
                #print("here3")
                all_syns = wn.synsets(word)
                for syn in all_syns:
                    #print(syn.pos())
                    if syn.pos() == pos:
                        return syn
    return False

# this functions take all the single and double legth phrases form grand_list and gets sysnset for all them. (1 each)
def get_singles_synset(uni_list,pos_tag_text):
    single_synset =[]
    #get synsets of all singletons
    for singles in uni_list:
        singles_syn = get_synset(singles,pos_tag_text, 'n')
        if singles_syn:
            single_synset.append(singles_syn)    
    return single_synset

#get common parents
def get_lcs(uni_list,pos_tag_text):
    #get all relevant sysnsets
    all_synsets = get_singles_synset(uni_list,pos_tag_text)
    list_of_all_lcs =[]
    for syn in all_synsets:
        for syn2 in all_synsets[all_synsets.index(syn)+1:]:
            lcs = syn.lowest_common_hypernyms(syn2)
            if len(lcs)> 0:
                if lcs[0] not in list_of_all_lcs:
                    list_of_all_lcs.append(lcs[0])
    return list_of_all_lcs

# get themes
def get_theme(uni_list,pos_tag_text):
    # get common parent
    parent_sysnset = get_lcs(uni_list,pos_tag_text)
    # filter out absolute top level and get lemma_names
    lemma_names =[]
    for synset in parent_sysnset:
        if synset.min_depth() != 0:
            #print(synset)
            for each_name in synset.lemma_names():
                if each_name not in lemma_names:
                    lemma_names.append(each_name)
                break
    return lemma_names

def get_cluster_count(document):
    text = str(document['text'][0])
    cleantext = get_document_text(text)
    tagged_tokens = tag_my_text(tokenize_text_sent(cleantext))
    np_list = noun_phrase_finder(tagged_tokens)
    top_np = get_top_unigrams(np_list)
    themes = get_theme(top_np,tagged_tokens)
    return len(themes)

# Get the comments associated with a document
# Is the comment is less than 500 characters and references an attachment, discard the comment as
# the attachment has been processed as a separate comment.
def process_document(document):
    comments = []
    for c in document['comment_list']:
        c = c.replace('\n',' ')
        if 'attached' in c and len(c) < 500:
            pass
        elif len(c) <200:
            pass
        else:
            comments.append(str(c))
    return comments

# Tokenize and lower the text
# Discard tokens fewer than 3 characters as these are likely stopwords or artifacts
# from converting the PDF attacments to text.
# Modified from Brandon Rose
def tokenize_text_cluster(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in tokenize_text(sent)]
    filtered_tokens = []
    for token in tokens:
        if len(token) > 2:
            filtered_tokens.append(token)
    return filtered_tokens

# Vectorize the comments using a tfidf vectorizer
# ngram_range set to 2-3 as manual inspection of the results indicated ngrams of this
# length had more semantic value than unigrams.
# Min_df set to 0.18 to limit the number of lower frequency words from being included
# as features while still finding enough ngrams to use as features
def vectorize_comments(comments):
    tfidf_vec = TfidfVectorizer(tokenizer=tokenize_text_cluster,
                                stop_words='english',
                                ngram_range=(2,3),
                                min_df=0.18, max_df=0.9,
                                max_features=200000)
    tfidf_matrix = tfidf_vec.fit_transform(comments)
    return tfidf_matrix, tfidf_vec

# For each cluster, combine the comments into a single document
def mash_comments(all_comments):
    big_comment = []
    for cluster in all_comments:
        mashed = ""
        for comment in cluster:
            mashed += comment
            mashed += " "
        big_comment.append(mashed)
    return big_comment

# Return the top n differentiating ngrams for each comment cluster
# Runs tfidf on the 'mashed' comments to identify the differentiating ngrams
# Returns the ngrams with the highest tfidf scores for each cluster
# Adapted from: http://www.markhneedham.com/blog/2015/02/15/pythonscikit-learn-calculating-tfidf-on-how-i-met-your-mother-transcripts/
def top_words(all_comments, n_top_words):
    mashed_comments = mash_comments(all_comments)
    tfidf_matrix, tfidf_vec = vectorize_comments(mashed_comments)
    feature_names = tfidf_vec.get_feature_names()
    dense = tfidf_matrix.todense()
    top_words = []
    for i in range(0,len(mashed_comments)):
        cluster = dense[i].tolist()[0]
        word_scores = [pair for pair in zip(range(0, len(cluster)), cluster) if pair[1] > 0]
        sorted_word_scores = sorted(word_scores, key=lambda t: t[1] * -1)
        temp_top_words = []
        for word, score in [(feature_names[word_id], score) for (word_id, score) in sorted_word_scores][:n_top_words]:
            temp_top_words.append(word)
        top_words.append(temp_top_words)
    return top_words

# Clusters the comments for a document into num_clusters
# Returns a dictionary with the central comment for each cluster, all the other comments
# for each cluster, as well as the top ngrams for each cluster.
def cluster_comments(document, num_clusters):
    '''
    Clusters the comments for a document into num_clusters
    Returns a dictionary containing:
    - the central comment
    - all comments
    - the top n differentiating ngrams
    for each cluster
    '''
    cluster_dict = {}

    comments = process_document(document)
    tfidf_matrix, tfidf_vec = vectorize_comments(comments)
    
    # Run kmeans on the clusters
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    
    # Identify the center of each cluster and calculate the distance between each comment
    # in a cluster and the respective cluster center
    cluster_center_list = []
    for c in clusters:
        cluster_center_list.append(km.cluster_centers_[c])
    center_distances = paired_distances(tfidf_matrix, cluster_center_list)
    
    # Create a datafram of the comments, the comments' clusters, and their respective distances
    # from the cluster centers
    comment_clusters = {'comment': comments, 'cluster': clusters, 'dist': center_distances}
    comment_frame = pd.DataFrame(comment_clusters, index = [clusters] , columns = ['comment', 'cluster', 'dist'])
    
    # Create two lists:
    # one of the central comments for each cluster and
    # one of all the comments in each cluster
    central_comments = []
    all_comments = []
    for i in range(num_clusters):
        central_comments.append(comment_frame[comment_frame.cluster==i].min().comment)
        all_comments.append(list(comment_frame[comment_frame.cluster==i]['comment']))
    
    # Find the top 6 differentiating ngrams for each cluster
    tfidf_words = top_words(all_comments, 6)
    
    # Build the dictionary
    cluster_dict['central_comments'] = central_comments
    cluster_dict['all_comments'] = all_comments
    cluster_dict['top_words'] = tfidf_words
    
    return cluster_dict

data = []

doc_list1 = load(open("data/Master_doc_content",'rb'))
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
    document = doc_list1[i]
    cluster_num = get_cluster_count(document)
    if cluster_num < 2:
        cluster_num = 2
    clust_dict = cluster_comments(document, cluster_num)
    clust_dict["doc_id"], clust_dict["doc_title"] = doc_id1[i], doc_title1[i]
    data.append(clust_dict)

for i in range(len(doc_list2)):
    print(i)
    document = doc_list2[i]
    cluster_num = get_cluster_count(document)
    if cluster_num < 2:
        cluster_num = 2
    clust_dict = cluster_comments(document, cluster_num)
    clust_dict["doc_id"], clust_dict["doc_title"] = doc_id2[i], doc_title2[i]
    data.append(clust_dict)

top_obj = {}
top_obj["data"] = data

with open('data/comment_data.json', 'w') as outfile:
    json.dump(top_obj, outfile)

# Library imports
from pickle import dump, load
import nltk
from nltk import word_tokenize,FreqDist
import re
from nltk.corpus import wordnet as wn
from itertools import combinations

def get_document_text(raw_text):
    """ This function takes in raw document text as input which we receive from the API and returns a clean text 
    of the associated document. It cleans up any HTML code in the text, newline characters, and extracts supplemental
    information part of the document.
    
    INPUT: string
    OUTPUT: string
    """
    raw_text = raw_text.replace('\n',' ')
    raw_text = raw_text.replace('*','') # added
    raw_text = raw_text.replace('\r',' ') # added
    supp_info_idx = raw_text.find("SUPPLEMENTARY INFORMATION:")
    summary_idx = raw_text.find("SUMMARY:")
    dates_idx = raw_text.find("DATES:")
    suppl_info = raw_text[supp_info_idx+26:] # To leave out the string 'Supplementary Information'
    summary = raw_text[summary_idx+8:dates_idx]
    # Remove any residual HTML tags in text
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', suppl_info)
    cleansummary = re.sub(cleanr, '', summary)
    return cleantext, cleansummary

def tokenize_text(corpus):
    pattern = r'''(?x)    # set flag to allow verbose regexps
    (([A-Z]\.)+)       # abbreviations, e.g. B.C.
    |(\w+([-']\w+)*)       # words with optional internal hyphens e.g. after-ages or author's
    '''
    tokens = nltk.regexp_tokenize(corpus,pattern)
    all_token = [word.lower() for token in tokens for word in token if word != "" 
                 and word[0] != "'" and word[0] != "-"]
    return all_token

def tokenize_text_sent(corpus):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sents = sent_tokenizer.tokenize(corpus) # Split text into sentences    
    return [tokenize_text(sent) for sent in raw_sents]

def tag_my_text(sents):
    return [nltk.pos_tag(sent) for sent in sents]

pos_tagged_collection = tag_my_text(tokenize_text_sent(doc_text))

def extract_all_nouns(pos_tagged):
    # We need to extract all nouns in the text - meaning throw out anything that's not a noun
    noun_tags = []
    for idx,sent in enumerate(pos_tagged):
        all_nouns = [item for item in sent if item[1][0] == 'N']
        if len(all_nouns) > 0:
            noun_tags.append(all_nouns)
    
    return noun_tags

nouns_text = extract_all_nouns(pos_tagged_collection)

# Lesk algorith for disambiguation in case of multiple synsets of a word
def compare_overlaps_greedy(context, synsets_signatures, pos=None):
    """
    Calculate overlaps between the context sentence and the synset_signature
    and returns the synset with the highest overlap.
    
    :param context: ``context_sentence`` The context sentence where the ambiguous word occurs.
    :param synsets_signatures: ``dictionary`` A list of words that 'signifies' the ambiguous word.
    :param pos: ``pos`` A specified Part-of-Speech (POS).
    :return: ``lesk_sense`` The Synset() object with the highest signature overlaps.
    """
    # if this returns none that means that there is no overlap
    max_overlaps = 0
    lesk_sense = None
    for ss in synsets_signatures:
        if pos and str(ss.pos()) != pos: # Skips different POS.
            continue
        overlaps = set(synsets_signatures[ss]).intersection(context)
        if len(overlaps) > max_overlaps:
            lesk_sense = ss
            max_overlaps = len(overlaps)  
    return lesk_sense

def lesk(context_sentence, ambiguous_word, pos=None, dictionary=None):
    """
    This function is the implementation of the original Lesk algorithm (1986).
    It requires a dictionary which contains the definition of the different
    sense of each word. See http://goo.gl/8TB15w

        >>> from nltk import word_tokenize
        >>> sent = word_tokenize("I went to the bank to deposit money.")
        >>> word = "bank"
        >>> pos = "n"
        >>> lesk(sent, word, pos)
        Synset('bank.n.07')
    
    :param context_sentence: The context sentence where the ambiguous word occurs.
    :param ambiguous_word: The ambiguous word that requires WSD.
    :param pos: A specified Part-of-Speech (POS).
    :param dictionary: A list of words that 'signifies' the ambiguous word.
    :return: ``lesk_sense`` The Synset() object with the highest signature overlaps.
    """
    if not dictionary:
        dictionary = {}
        for ss in wn.synsets(ambiguous_word):
            dictionary[ss] = ss.definition().split()
    best_sense = compare_overlaps_greedy(context_sentence, dictionary, pos)
    return best_sense
    #return dictionary 

# this function takes in a word and gets the most relevant synset based on context from the text. 
# for exact algorith refer the text above ("what I want to do" markdown)
def get_synset(word,sentence ,pos):
    if len(wn.synsets(word)) == 1:
        #print("here1")
        return wn.synsets(word)[0]
    else:
#         #get all context sentences
#         all_sent =[]
#         for sent in pos_tag_text:
#             for (w,t) in sent:
#                 if w == word:
#                     all_sent.append(sent)
        #call lesk here
        app_syn = lesk(sentence, word, pos)
        if app_syn != None:
            #print("here2")
            return app_syn
        else:
            #give up and choose 1st synset from list with matching pos
            #print("here3")
            all_syns = wn.synsets(word)
            for syn in all_syns:
                #print(syn.pos())
                if syn.pos() == pos:
                    return syn
    return False

def sent_distance(sent1,sent2):
    n = len(sent1) # sent1 and sent2 are lists here
    cum_sum = 0
    for i in range(n):
        word_dist = word_sent_dist(sent1[i],sent1,sent2)
        cum_sum += word_dist
    return (1/n) * cum_sum

def word_sent_dist(w,s1,s2):
    sem_dist = []
    for word in s2:
        wordDistance = words_dist(w,s1,word,s2)
        sem_dist.append(wordDistance)
    if len(sem_dist) > 0: 
        return min(sem_dist) 
    else: 
        return 0

def words_dist(w1,s1,w2,s2):
    syn_w1 = get_synset(w1[0],s1,'n')
    syn_w2 = get_synset(w2[0],s2,'n')
    
    if (not syn_w1) or (not syn_w2):
        return 2
    
    dca = syn_w1.lowest_common_hypernyms(syn_w2)

    if len(dca) == 0: # In case we don't find a lowest common hypernym
        return 2 # Since now w1_dca = w1_root and w2_dca = w2_root
    
    w1_dca = hyp_dist(syn_w1,dca[0])
    w1_root = hyp_dist(syn_w1)
    w1_root = w1_root if w1_root > 0 else 1
    
    w2_dca = hyp_dist(syn_w2,dca[0])
    w2_root = hyp_dist(syn_w2)
    w2_root = w2_root if w2_root > 0 else 1
    
    return (w1_dca/w1_root) + (w2_dca/w2_root)

def hyp_dist(syn_word,ancestor=None):
    if ancestor:
        depth1 = syn_word.min_depth()
        depth2 = ancestor.min_depth()
        
        root_depth = depth1 - depth2
    else:
        root_depth = syn_word.min_depth()
    
    return root_depth

sent_distance(nouns_text[0],nouns_text[1])

def create_sem_dist_matrix(corpus):
    dist_mat = []
    pos_tagged_collection = tag_my_text(tokenize_text_sent(doc_text))
    nouns_text = extract_all_nouns(pos_tagged_collection)
    
    sent_combinations = combinations(nouns_text,2)
    for combo in combinations(nouns_text,2):
        
        # Pair combinations of sentences are here, calculate semantic distance between them
        sem_dist = sent_distance(combo[0],combo[1])
        dist_mat.append(sem_dist)
        
    return dist_mat

res = create_sem_dist_matrix(doc_text)

