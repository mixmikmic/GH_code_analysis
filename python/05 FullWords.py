import psycopg2
from psycopg2.extensions import register_adapter, AsIs
from collections import Counter
import pandas as pd
import numpy as np
from time import time
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
np.set_printoptions(suppress=True,precision=10)

def obtainSubjects(dbname="mimic") :
    conn = psycopg2.connect("dbname="+dbname)
    cur = conn.cursor()
    select_stament = ("SELECT DISTINCT subject_id "
                      " FROM waveformqrst WHERE centroid IS NOT NULL" # limit 150"
    )
    cur.execute(select_stament)
    subject = []
    for row in cur :
        subject.append(row[0])
    conn.close()
    return subject

def obtainWord(subject,dbname="mimic") :
    conn = psycopg2.connect("dbname="+dbname)
    cur = conn.cursor()
    select_stament = ("SELECT centroid "
                      " FROM waveformqrst WHERE subject_id="+str(subject)+" ORDER BY qrtsorder"
    )
    cur.execute(select_stament)
    centroids = ""
    for row in cur :
        centroid = row[0]
        if centroid is not None :
            centroids= centroids+centroid
    conn.close()
    return centroids

def deleteWord(dbname="mimic") :
    conn = psycopg2.connect("dbname="+dbname)
    cur = conn.cursor()
    select_stament = "DELETE FROM subjectwords"
    cur.execute(select_stament)
    conn.commit()
    cur.close()
    conn.close()

def instertWord(words,dbname="mimic") :
    conn = psycopg2.connect("dbname="+dbname)
    cur = conn.cursor()
    insert_statement = 'INSERT into subjectwords (%s) values %s'
    columns = words.keys()
    values = [words[column] for column in columns]
#    print(cur.mogrify(insert_statement, (AsIs(','.join(columns)), tuple(values))))
    cur.execute(insert_statement, (AsIs(','.join(columns)), tuple(values)))
    conn.commit()
    cur.close()
    conn.close()

def selectWord(dbname="mimic") :
    conn = psycopg2.connect("dbname="+dbname)
    cur = conn.cursor()
    select_stament = ("SELECT subject_id,word "
                      " FROM subjectwords WHERE length(word)>1000"
    )
    cur.execute(select_stament)
    select = []
    for row in cur :
        select.append([row[0],row[1]])
    conn.close()
    return select

deleteWord()
subjects = obtainSubjects() 
print(len(subjects))
for subject in subjects :
    word = obtainWord(subject)
    if word is not None:
        words = {'subject_id':subject,'word':word}
        print(len(word),end=":")
        instertWord(words)
        print(words['subject_id'],end=",")

def get_all_substrings(input_string,length=5):
    substrings = []
    for j in range(len(input_string)) :
        for i in range(length) :
            substrings.append(input_string[j:j+i+1])
    return Counter(substrings)

def existMatrix(word,subject,dbname="mimic") :
    conn = psycopg2.connect("dbname="+dbname)
    cur = conn.cursor()
    select_stament = ("SELECT 1 "
                      " FROM matrix WHERE subject_id='"+str(subject)+"' AND word='"+str(word)+"'"
    )
    cur.execute(select_stament)
    exist = False
    for row in cur :
        exist = True
    cur.close()
    conn.close()
    return exist

def saveMatrix(matrix,dbname="mimic") :
    conn = psycopg2.connect("dbname="+dbname)
    cur = conn.cursor()
    insert_statement=('INSERT INTO matrix(word,subject_id,counting)'
                      ' SELECT unnest( %(word)s ) ,'
                      ' unnest( %(subject_id)s) ,'
                      ' unnest( %(counting)s)')
    word=[r['word'] for r in matrix]
    subject_id=[r['subject_id'] for r in matrix]
    counting=[r['counting'] for r in matrix]
#    print(cur.mogrify(insert_statement,locals()))
    cur.execute(insert_statement,locals())
    conn.commit()
    cur.close()
    conn.close()

def cleanMatrix(dbname="mimic") :
    conn = psycopg2.connect("dbname="+dbname)
    cur = conn.cursor()
    select_stament = ("DELETE "
                      " FROM matrix"
    )
    cur.execute(select_stament)
    conn.commit()
    cur.close()
    conn.close()

words = selectWord()
cleanMatrix()
i=0
for word in words :
    subject = word[0]
    subs =get_all_substrings(word[1],length=10)
    matrix = []
    for key in subs:
        matrix.append({'word':key,'counting':subs[key],'subject_id':subject})
#        if not existMatrix(key,subject) :
    if matrix != [] :
        print(i,end=",")
        i=i+1
        saveMatrix(matrix)
#    print(subs.keys())

def selectMatrix(dbname="mimic") :
    conn = psycopg2.connect("dbname="+dbname)
    cur = conn.cursor()
    select_stament = ("SELECT subject_id,word,counting "
                      " FROM matrix ORDER BY subject_id"
    )
    cur.execute(select_stament)
    select = []
    for row in cur :
        select.append((row))
    cur.close()
    conn.close()
    return select

labels = ['subject_id', 'Word', 'Counting']
df = pd.DataFrame.from_records(selectMatrix(), columns=labels)
table = pd.pivot_table(df,index=["subject_id"],columns=["Word"],values=["Counting"],
                       aggfunc={"Counting":[np.sum]},fill_value=0)
print(table)

# Fit the NMF model
t0 = time()
nmf = NMF(n_components=30, random_state=1,alpha=.1, l1_ratio=.5)
W = nmf.fit_transform(table)
H = nmf.components_
print(W)
print("done in %0.3fs." % (time() - t0))

print(np.shape(W))
print(np.shape(H))

def patientIsAlive(patient,dbname="mimic") :
    conn = psycopg2.connect("dbname="+dbname)
    cur = conn.cursor()
    select_stament = ("SELECT dod "
                      " FROM patients WHERE subject_id = "+str(patient)
    )
    cur.execute(select_stament)
    select = []
    for row in cur :
        select.append(1 if row[0] is not None else 0 )
    cur.close()
    conn.close()
    return select

patients = []
for patient in table.index :
    patients.append(patientIsAlive(patient)[0])
print(len(patients))

# flatten y into a 1-D array
y = np.ravel(patients)
modelo_lr = LogisticRegression()
modelo_lr.fit(y=y,X=W)

modelo_lr.score(W, y)

y.mean()

predicion = modelo_lr.predict(W)
print(W[:1])
print(predicion,y[:1])

modelo_lr

y



