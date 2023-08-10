import nltk
import os
import codecs
import string
import matplotlib.pyplot as plt
import matplotlib as mpl
from gensim import corpora, models, similarities #Latent Dirichlet Allocation implementation with Gensim
import pyLDAvis
import pyLDAvis.gensim
from IPython.display import clear_output

import getpass
from sshtunnel import SSHTunnelForwarder
import pymysql

import pickle

outputs = "../outputs/"

if os.name == 'nt':
    encoding_type = 'utf-8'
    ssh_priv_key = 'C:/Users/marcelo.ribeiro/Dropbox/A-Marcelo/Educação-Trabalho/2016-CPDOC/Remoto/marcelo_priv_rsa'
    ssh_user='marcelobribeiro'
    sql_user='marcelobribeiro'
else:
    encoding_type = 'ISO-8859-1'
    ssh_priv_key = '/home/rsouza/.ssh/id_rsa'
    ssh_user='rsouza'
    sql_user='rsouza'

count = 0
texts = []
pass_sshkey = getpass.getpass()
pass_mysql = getpass.getpass()

with SSHTunnelForwarder(('200.20.164.146', 22),
                        ssh_private_key=ssh_priv_key,
                        ssh_private_key_password=pass_sshkey,
                        ssh_username=ssh_user,
                        remote_bind_address=('127.0.0.1', 3306)) as server:
    
    conn = pymysql.connect(host='localhost', 
                           port=server.local_bind_port, 
                           user=sql_user,
                           passwd=pass_mysql,
                           db='CPDOC_AS',
                           use_unicode=True, 
                           charset="utf8")
    cur = conn.cursor()
    
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    selects texts from mysql database to start topic modeling
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    cur.execute("SELECT * FROM CPDOC_AS.docs WHERE main_language = 'pt' AND (readability > 0.4 OR readability = -1) ") # filtra textos   
    data = cur.fetchall()
    numrows = cur.rowcount
    percentil = numrows/100
    
    for row in data:
        count += 1
        if row is None: break

        if count % percentil == 0: 
            clear_output()
            print(int(count/percentil),'% done')

        text =  row[4]
        text = text.split()
        symbols = [x for x in string.punctuation]
        text = [p for p in text if p not in symbols]
        text = [p.strip(string.punctuation) for p in text]
        text = [p for p in text if not p.isdigit()]
        text = [p for p in text if len(p)>1]
        texts.append(text)

additional_words = ['mr','one', 'two', 'three', 'four', 
                    'five', 'um', 'dois', 'três', 'quatro', 
                    'cinco', 'janeiro', 'fevereiro', 'março', 
                    'abril', 'maio', 'junho', 'julho', 'agosto', 
                    'setembro', 'outubro', 'novembro', 'dezembro', 
                    'january', 'february', 'march', 'april', 'may', 
                    'june', 'july', 'august', 'september', 
                    'october', 'november', 'december', 'países', 
                    'ser', 'país', 'ainda', 'milhões', 'maior', 
                    'anos', 'grande', 'apenas', 'outros', 'pode', 
                    'parte', 'partes', 'item', 'vossa', 'nota', 
                    'havia', 'pt', 'vg', 'ptvg', 'eh', 'nr', 'hrs', 
                    'pais', 'parte', 'hoje', 'brasemb', 'ontem', 
                    'dia', 'countries', 'would', 'new', 'also', 
                    'must', 'draft', 'shall', 'item', 'page', 
                    'th', 'anos', 'ii', 'dias', 'poderá', 'caso', 
                    'casos', 'qualquer', 'ano', 'mil', 'pessoas', 
                    'único', 'única', 'únicos', 'únicas', 'índice', 
                    'expedido', 'co', 'mm', 'er', 'via', 'ww', 'ra', 
                    'ia', 'ca', 'nu', 'wa', 'aa', 'ms', 'dc', 'mmm', 'pa']

stopwords = nltk.corpus.stopwords.words('english') +             nltk.corpus.stopwords.words('portuguese') +             nltk.corpus.stopwords.words('french') +             nltk.corpus.stopwords.words('spanish') +             nltk.corpus.stopwords.words('german') +             additional_words

stopwords = list(set(stopwords))

get_ipython().magic('time texts = [[word for word in text if word not in stopwords] for text in texts]')

print(len(texts[0]))

dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=1, no_above=0.8)
dictionary.filter_tokens(bad_ids=[0,]) #retira palavras a partir do id
corpus = [dictionary.doc2bow(text) for text in texts]

len(corpus)

file_corpus = '../outputs/LDAcorpus.pkl'
file_dictionary = '../outputs/LDAdictionary.pkl'

''' caso queira carregar os arquivos '''
corpus = pickle.load(open(file_corpus, 'rb'))
dictionary = pickle.load(open(file_dictionary, 'rb'))

get_ipython().magic('time lda45 = models.LdaModel(corpus, num_topics=45, id2word=dictionary, passes=50, eval_every=1, random_state=0)')
pickle.dump(lda45, open('../outputs/model_lda_45.pkl', 'wb'))

lda45.print_topics(-1, num_words=5)

pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(lda45, corpus, dictionary)

data_ldavis = pyLDAvis.gensim.prepare(lda45, corpus, dictionary)
pyLDAvis.save_html(data_ldavis, os.path.join(outputs,'pyldavis_output_45topics.html'))

get_ipython().magic('time lda60_00 = models.LdaModel(corpus, num_topics=60, id2word=dictionary, passes=50, eval_every=1, random_state=0)')
pickle.dump(lda60_00, open('../outputs/model_lda_60_rs_00.pkl', 'wb'))

pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(lda60_00, corpus, dictionary)

data_ldavis = pyLDAvis.gensim.prepare(lda60_00, corpus, dictionary)
pyLDAvis.save_html(data_ldavis, os.path.join(outputs,'pyldavis_output_60topics_rs00.html'))

get_ipython().magic('time lda60_01 = models.LdaModel(corpus, num_topics=60, id2word=dictionary, passes=50, eval_every=1, random_state=1)')
pickle.dump(lda60_01, open('../outputs/model_lda_60_rs_01.pkl', 'wb'))

pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(lda60_01, corpus, dictionary)

data_ldavis = pyLDAvis.gensim.prepare(lda60_01, corpus, dictionary)
pyLDAvis.save_html(data_ldavis, os.path.join(outputs,'pyldavis_output_60topics_rs01.html'))

get_ipython().magic('time lda100_00 = models.LdaModel(corpus, num_topics=100, id2word=dictionary, passes=50, eval_every=1, random_state=0)')
pickle.dump(lda100_00, open('../outputs/model_lda_100_rs_00.pkl', 'wb'))

pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(lda100_00, corpus, dictionary)

data_ldavis = pyLDAvis.gensim.prepare(lda100_00, corpus, dictionary)
pyLDAvis.save_html(data_ldavis, os.path.join(outputs,'pyldavis_output_100topics_rs00.html'))

get_ipython().magic('time hdp = models.HdpModel(corpus, id2word=dictionary)')

hdp.print_topics()

pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(hdp, corpus, dictionary)

data_ldavis = pyLDAvis.gensim.prepare(hdp, corpus, dictionary)
pyLDAvis.save_html(data_ldavis, os.path.join(outputs,'pyldavis_output_hdp.html'))

