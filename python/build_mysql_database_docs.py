import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import re
import pymysql
import getpass

from sshtunnel import SSHTunnelForwarder
import time

import codecs
from langdetect import detect
from IPython.display import clear_output

def to_zero(x):
    if x == '': x = '0'
    return x
def special_sort(l):
    convert = lambda text: int(text) if text.isdigit() else str(text)
    alphanum_key = lambda key: [ convert(to_zero(c)) for c in filter(None, re.split('(\d)A|A\d|([A-Z]*)-A?|.txt', key))] 
    return sorted(l, key = alphanum_key)

if os.name == 'nt':
    encoding_type = 'utf-8'
    ssh_priv_key = 'C:/Users/marcelo.ribeiro/Dropbox/A-Marcelo/Educação-Trabalho/2016-CPDOC/Remoto/marcelo_priv_rsa'
    ssh_user='marcelobribeiro'
    sql_user='marcelobribeiro'
    path = "C:/Users/marcelo.ribeiro/Documents/textfiles-corrected-regrouped/"
    outputs = "C:/Users/marcelo.ribeiro/Documents/outputs/"
else:
    encoding_type = 'ISO-8859-1'
    ssh_priv_key = '/home/rsouza/.ssh/id_rsa'
    ssh_user='rsouza'
    sql_user='rsouza'
    path = "/home/rsouza/Documentos/text-learning-tools/textfiles-corrected-regrouped/"
    outputs = "/home/rsouza/Documentos/text-learning-tools/outputs/"

files = [f for f in sorted(os.listdir(path))]
fullpath_list = []
fullpath = ''
for file in files:
    fullpath = path+file
    fullpath_list.append(fullpath)
fullpath_list = special_sort(fullpath_list)

fullpath_list[0:10]

doc_class = []
lang_class = 'none'
count_doc = 0
not_readable = []
percentil = int(len(fullpath_list)/100)

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
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    cur.execute("DROP TABLE IF EXISTS docs")
    cur.execute('''CREATE TABLE IF NOT EXISTS docs 
               (id VARCHAR(31) PRIMARY KEY, dossie_id_draft VARCHAR(31), 
               main_language VARCHAR(10), readability DECIMAL(3,2), body LONGTEXT
               DEFAULT NULL)
               ENGINE=MyISAM DEFAULT CHARSET='utf8';''')
    
    ''' iterates through texts '''
    for txt in fullpath_list:
        
        ''' measures completion percentage '''
        count_doc += 1
        if count_doc % percentil == 0: print(int(count_doc/percentil),'% done')
        if count_doc % (percentil-1) == 0: clear_output()
        
        ''' captures info about date, year, month and ids '''
        txt_date = re.sub('.*(19\d\d\.\d\d\.\d\d).*', r'\1', txt)
        txt_year = re.sub('.*(19\d\d).*', r'\1', txt)
        txt_month = re.sub('.*19\d\d\.(\d\d).*', r'\1', txt)
        txt_id = re.sub('.*AAS_mre_(.*).txt', r'\1', txt)
        dossie_id = re.sub('.*AAS_mre_(.*)_doc.*', r'\1', txt)
        
        ''' makes analysis in each document '''
        with open(txt, 'r', encoding=encoding_type) as f:
            txt_body = f.read()
           
            ''' identifies main language and readability of each document '''
            text_split = re.split('\.|\?|\:|\,', txt_body)
            pt_count = en_count = es_count = fr_count = de_count = lang_count = total_count = 0
            for phrase in text_split:
                try: 
                    if len(re.findall("[^\W\d]", phrase)) <= 10: continue
                    language = detect(phrase)
                    total_count += 1
                except: 
                    continue
                if language == 'pt':
                    pt_count += 1
                if language == 'en':
                    en_count += 1
                if language == 'es':
                    es_count += 1
                if language == 'fr':
                    fr_count += 1
                if language == 'de':
                    de_count += 1
            lang_count = pt_count + en_count + es_count + fr_count + de_count        
            if total_count == 0: readability_ratio = 0
            else: readability_ratio = float(lang_count/total_count)
            if readability_ratio < 0.3: 
                not_readable.append(txt)
                continue
            elif total_count > 10: readability = readability_ratio
            else: readability = -1
            ''' note: with the criteria, documents might have readability but no lang_class '''
            if de_count/total_count > 0.3 and de_count >= 3: 
                lang_class = 'de'
            if fr_count/total_count > 0.3 and fr_count >= 3: 
                lang_class = 'fr'
            if es_count/total_count > 0.3 and es_count >= 3: 
                lang_class = 'es'
            if en_count/total_count > 0.3 and en_count >= 3: 
                lang_class = 'en'
            if pt_count/total_count > 0.3 and pt_count >= 3: 
                lang_class = 'pt'
            
            ''' inserts data into mysql '''
            query = "INSERT INTO docs VALUES (%s, %s, %s, %s, %s)"
            cur.execute(query, (txt_id, dossie_id, lang_class, readability, txt_body))

not_readable_files = os.path.join(outputs, "not_readable_files.txt")

with open(not_readable_files, 'w+', encoding='utf-8') as f: 
    text = f.write(not_readable[0])
    text = f.write('\r\n')
for file in not_readable[1:]:
    with open(not_readable_files, 'a+', encoding='utf-8') as f:
        text = f.write(file)
        text = f.write('\r\n')



