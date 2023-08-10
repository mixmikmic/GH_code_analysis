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
import pymysql

import pickle

sql_user='marcelobribeiro'
path_outputs = '../../text-learning-tools/outputs/'

file_corpus = path_outputs+'LDAcorpus.pkl'
file_dictionary = path_outputs+'LDAdictionary.pkl'
file_lda = path_outputs+'model_lda_100_rs_00.pkl'

corpus = pickle.load(open(file_corpus, 'rb'))
dictionary = pickle.load(open(file_dictionary, 'rb'))
lda = pickle.load(open(file_lda, 'rb'))

len(corpus)

lda.print_topics(-1, num_words=5)

# Caso queira trabalhar apenas com os tópicos mais coesos
main_topics_list = [99, 61, 39, 69, 63, 19, 17, 82, 56, 42, 94, 18, 89, 4, 49, 7, 32, 25, 78, 15, 73, 45, 27, 16, 97, 47, 26, 
                   58, 59, 36, 10, 57, 86, 34, 28, 53, 12, 37, 51, 76, 83, 93, 46, 31, 11, 13, 65]

pass_sshkey = getpass.getpass()
pass_mysql = getpass.getpass()
with SSHTunnelForwarder(('200.20.164.146', 22),
                        ssh_private_key = ssh_priv_key,
                        ssh_private_key_password = pass_sshkey,
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
    inserts data into mysql database
    captures documents from docs table
    creates topic-doc table
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    cur.execute("DROP TABLE IF EXISTS topic_doc")
    cur.execute('''CREATE TABLE IF NOT EXISTS topic_doc
               (doc_id VARCHAR(31), topic_id smallint(6), topic_score FLOAT
               DEFAULT NULL)
               ENGINE=MyISAM DEFAULT CHARSET='utf8';''')

    cur.execute("SELECT * FROM CPDOC_AS.docs WHERE main_language = 'pt' AND (readability > 0.4 OR readability = -1) ")  
    data = cur.fetchall()
    numrows = cur.rowcount
    percentil = numrows/100
    
    for row in data:
        
        ### mede percentual de conclusão da tarefa ###
        count = data.index(row)
        if count % 100 == 0: 
            clear_output()
            print(int(count/percentil),'% done')
        
        text =  row[4]
        text = text.split()
        text_bow = dictionary.doc2bow(text)
        score_list = lda[text_bow]
        doc_id = row[0]
        for score in score_list:
            topic_id = str(score[0])
            topic_score = str(score[1])
            #if topic_id not in main_topics_list: continue
            query = "INSERT INTO topic_doc VALUES (%s, %s, %s)"
            cur.execute(query, (doc_id, topic_id, topic_score))
            #print(doc_id, topic_id, topic_score)   
    cur.execute("ALTER TABLE CPDOC_AS.topic_doc ORDER BY topic_id ASC, topic_score DESC")

topics_list = [['International Cooperation for Development', 99], ['Geisel foreign policy: ideas and action', 61], 
               ['Brazilian government and private investment', 39], ['UN system', 63], 
               ['International Economic Relations of Brazil', 56], ['United States of America', 89], 
               ['Latin America and Caribbean', 4], ['Itaipu plant: technical discussions', 49], ['Nuclear Brazil', 7], 
               ['Brazil, Africa and decolonization', 45]]
topics_id_list = [i[1] for i in topics_list]
#for i in range(0,99): print(i)

topics_id_list

pass_sshkey = getpass.getpass()
pass_mysql = getpass.getpass()
with SSHTunnelForwarder(('200.20.164.146', 22),
                        ssh_private_key = ssh_priv_key,
                        ssh_private_key_password = pass_sshkey,
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
    inserts data into mysql database
    captures documents from docs table
    creates topic-doc table
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    cur.execute("DROP TABLE IF EXISTS topics")
    cur.execute('''CREATE TABLE IF NOT EXISTS topics
               (id SMALLINT(6) PRIMARY KEY, title MEDIUMTEXT, name VARCHAR(250)
               DEFAULT NULL)
               ENGINE=MyISAM DEFAULT CHARSET='utf8';''')
    topic_title = ''
    for topic in topics_list:
        topic_name = topic[0]
        #topic_title = topic[1]
        topic_id = topic[1]
        
        query = "INSERT INTO topics VALUES (%s, %s, %s)"
        cur.execute(query, (topic_id, topic_title, topic_name))
    
    topic_name = ''
    for i in range(0,99): 
        if i not in topics_id_list:
            query = "INSERT INTO topics VALUES (%s, %s, %s)"
            cur.execute(query, (i, topic_title, topic_name))
    cur.execute("ALTER TABLE CPDOC_AS.topics ORDER BY id ASC")



