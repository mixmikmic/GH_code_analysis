from nltk.stem.wordnet import WordNetLemmatizer

from gensim import models
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import STOPWORDS

import string 
import codecs

spe_char = {u'β': 'beta', u'α': 'alpha', u'µm': 'micron'}

conrad2013 = """The inability of targeted BRAF inhibitors to produce long-lasting improvement in the clinical outcome of melanoma highlights a need to identify additional approaches to inhibit melanoma growth. 
            Recent studies have shown that activation of the Wnt/β-catenin pathway decreases tumor growth and cooperates with ERK/MAPK pathway inhibitors to promote apoptosis in melanoma. 
            Therefore, the identification of Wnt/β-catenin regulators may advance the development of new approaches to treat this disease. 
            In order to move towards this goal we performed a large scale small-interfering RNA (siRNA) screen for regulators of β-catenin activated reporter activity in human HT1080 fibrosarcoma cells. 
            Integrating large scale siRNA screen data with phosphoproteomic data and bioinformatics enrichment identified a protein, FAM129B, as a potential regulator of Wnt/β-catenin signaling.  
            Functionally, we demonstrated that siRNA-mediated knockdown of FAM129B in A375 and A2058 melanoma cell lines inhibits WNT3A-mediated activation of a β-catenin-responsive luciferase reporter and inhibits expression of the endogenous Wnt/β-catenin target gene, AXIN2. 
            We also demonstrate that FAM129B knockdown inhibits apoptosis in melanoma cells treated with WNT3A. These experiments support a role for FAM129B in linking Wnt/β-catenin signaling to apoptosis in melanoma.
            The incidence of melanoma continues to rise across the U.S. at a rate faster than any other cancer. Malignant melanoma has a poor prognosis with a 5-year survival rate of only 15%3. 
            The recently approved therapeutic, vemurafenib, extends median patient survival by 7 months. This major advance raises expectations that even greater rates of survival might be attainable with combination therapies."""

huang2015 = """Mouse GnT1IP-L, and membrane-bound GnT1IP-S (MGAT4D) expressed in cultured cells inhibit MGAT1, the N-acetylglucosaminyltransferase that initiates the synthesis of hybrid and complex N-glycans. 
            However, it is not known where in the secretory pathway GnT1IP-L inhibits MGAT1, nor whether GnT1IP-L inhibits other N-glycan branching N-acetylglucosaminyltransferases of the medial Golgi. We show here that the luminal domain of GnT1IP-L contains its inhibitory activity. 
            Retention of GnT1IP-L in the endoplasmic reticulum (ER) via the N-terminal region of human invariant chain p33, with or without C-terminal KDEL, markedly reduced inhibitory activity. 
            Dynamic fluorescent resonance energy transfer (FRET) and bimolecular fluorescence complementation (BiFC) assays revealed homomeric interactions for GnT1IP-L in the ER, and heteromeric interactions with MGAT1 in the Golgi. 
            GnT1IP-L did not generate a FRET signal with MGAT2, MGAT3, MGAT4B or MGAT5 medial Golgi GlcNAc-tranferases. 
            GnT1IP/Mgat4d transcripts are expressed predominantly in spermatocytes and spermatids in mouse, and are reduced in men with impaired spermatogenesis."""

def parse_text(text):
    "Gets a text and outputs a list of strings."
    doc = [text.replace(unicode(i), spe_char.get(i)) for i in text if i in spe_char.keys()] or [text]
    return doc

conrad2013_parsed = parse_text(conrad2013)

huang2015_parsed = parse_text(huang2015)

def parse_from_file(text_file):
    "Retrieves the content of a text file"
    with codecs.open(text_file, mode='r', encoding='utf-8') as f:
        reader = f.read()
        return parse_text(reader)

def get_tokens(text_parsed):
    "Gets a text and retrieves tokens."
    # Tokenisation
    texts = text_parsed[0].lower().replace('\n', ' ').replace('/', ' ').replace('-', ' ').split(' ')
    # Remove punctuation and stop words
    tokens = [filter(lambda x:x not in string.punctuation, i)
               for i in texts if i != '' and i not in STOPWORDS]
    tokens_cleaned = [i for i in tokens if len(i) > 2 and not i.isdigit()]
    # print len(tokens_cleaned), [len(txt) for txt in tokens_cleaned]
    #return tokens_cleaned
    return tokens_cleaned

conrad2013_tokens = get_tokens(conrad2013_parsed)

huang2015_tokens = get_tokens(huang2015_parsed)

print conrad2013_tokens

def lemmatize_tokens(tokens):
    "Gets tokens and retrieves lemmatised tokens."
    # Lemmatisation using nltk lemmatiser
    lmtzr = WordNetLemmatizer()
    lemma = [lmtzr.lemmatize(t) for t in tokens]
    return lemma

conrad2013_lemma = lemmatize_tokens(conrad2013_tokens)

huang2015_lemma = lemmatize_tokens(huang2015_tokens)

print conrad2013_lemma

docs = [conrad2013_lemma, huang2015_lemma] # testing corpus

def bag_of_words(lemma):
    "Takes in lemmatised words and returns a bow."
    # Create bag of words from dictionnary
    dictionary = Dictionary(lemma)
    dictionary.save('text.dict')
    # Term frequency–inverse document frequency (TF-IDF)
    bow = [dictionary.doc2bow(l) for l in lemma] # Calculates inverse document counts for all terms
    return (bow, dictionary)

bow, dictionary = bag_of_words(docs)

print "Bag of words representation of the first document (tuples are composed by token_id and multiplicity):\n", bow[0]
print
for i in range(5):
    print "In the document, topic_id %d (word %s) appears %d time[s]" %(bow[0][i][0], docs[0][bow[0][i][0]], bow[0][i][1])
print "..."

def lda_model(dictionary, bow):
    ldamodel = models.ldamodel.LdaModel(bow, num_topics=10, id2word=dictionary, passes=5)
    return ldamodel

lda = lda_model(dictionary, bow)

for index, score in sorted(lda[bow[1]], key=lambda tup: -1*tup[1]):
    print "Score: %f\nTopic: %s" %(score, lda.print_topic(index, 10))

burns2015 = """Duplication of the yeast centrosome (called the spindle pole body, SPB) is thought to occur through a series of discrete steps that culminate in insertion of the new SPB into the nuclear envelope (NE). 
            To better understand this process, we developed a novel two-color structured illumination microscopy with single-particle averaging (SPA-SIM) approach to study the localization of all 18 SPB components during duplication using endogenously expressed fluorescent protein derivatives. 
            The increased resolution and quantitative intensity information obtained using this method allowed us to demonstrate that SPB duplication begins by formation of an asymmetric Sfi1 filament at mitotic exit followed by Mps1-dependent assembly of a Spc29- and Spc42-dependent complex at its tip. 
            Our observation that proteins involved in membrane insertion, such as Mps2, Bbp1, and Ndc1, also accumulate at the new SPB early in duplication suggests that SPB assembly and NE insertion are coupled events during SPB formation in wild-type cells."""

burns2015_tokens = lemmatize_tokens(get_tokens(parse_text(burns2015)))

bow_vector = dictionary.doc2bow(burns2015_tokens)
for index, score in sorted(lda[bow_vector], key=lambda tup: -1*tup[1]):
    print "Score: %f\n Topic: %s" %(score, lda.print_topic(index, 5))

print "Log perplexity of the model is", lda.log_perplexity(bow)



