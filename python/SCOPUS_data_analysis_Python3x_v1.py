import sqlite3
import matplotlib.pyplot as plt
import operator

sqlite_database = '/home/neel/scopus_data/scopus_data.sqlite'
conn = sqlite3.connect(sqlite_database)
c = conn.cursor()
c.execute("SELECT `Title`,`cited_rank` FROM `AI_scopus` ORDER BY `cited_rank` DESC LIMIT 0, 20;")
data = c.fetchall()
conn.close()
top_paper = {}
#print(data)

for x in data:
    text = (((str(x).replace("'","")).replace("(","")).replace(")",""))
    lis = text.split(",")
    #print(lis[0])
    #print(lis[1].strip())
    top_paper[str(lis[0])]= int(lis[1])
#print(top_paper)
plt.barh(range(len(top_paper)),top_paper.values(),align='center')
plt.yticks(range(len(top_paper)),list(top_paper.keys()))
plt.xlabel('\n Paper cited ')
plt.title("Top 20 Indian researcher's paper in SCOPUS journal \nfrom 2000 to 2016\n")
plt.ylabel('---- Paper ---- \n')
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 15
plt.rcParams["figure.figsize"] = fig_size
plt.show()

import sqlite3
import matplotlib.pyplot as plt
import operator

data = []
sqlite_database = '/home/neel/scopus_data/scopus_data.sqlite'
conn = sqlite3.connect(sqlite_database)
c = conn.cursor()
c.execute("SELECT `Authors`,`cited_rank` FROM `AI_scopus` ORDER BY `cited_rank` DESC LIMIT 0, 20;")
data = c.fetchall()
top_author = {}
text = str(data[0]).replace("'","")
for x in data:
    cite = (str(x)[-4:-1]).strip()
    authors = (str(x)[2:len(x)-7]).replace("'","")
    top_author[str(authors)] = int(cite)
    
#print(top_author)
conn.close()


plt.barh(range(len(top_author)),top_author.values(),align='center')
plt.yticks(range(len(top_author)),list(top_author.keys()))
plt.xlabel('\n Author cited ')
plt.title('Top 20 Indian researcher in SCOPUS journal\n from 2000 to 2016\n')
plt.ylabel('---- Authors ---- \n')
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 15
plt.rcParams["figure.figsize"] = fig_size
plt.show()

import sqlite3
import matplotlib.pyplot as plt
import operator

data = []
sqlite_database = '/home/neel/scopus_data/scopus_data.sqlite'
conn = sqlite3.connect(sqlite_database)
c = conn.cursor()
c.execute("SELECT `Year`,`IndexKeywords` FROM `AI_scopus` DESC LIMIT 0, 5000;")
data = c.fetchall()
#text = str(data[0])
#print(text[2:6])
#print(text[10:len(text)-2])
#tr = []
#tr = (text[10:len(text)-2]).split(";")
#print(tr)
conn.close()
data_dic = {}

z = 0 
word_lis = []
while z < len(data):
    text = str(data[z])
    year = str(text[2:6])
    #print(year)
    lis_word = (text[10:len(text)-2].replace(" ","")).split(";")
    #print(lis_word)
    if year == '2016':
        for word in lis_word:
            try:
                data_dic[str(word)] = int(data_dic[str(word)]) + 1
            except:
                data_dic[str(word)] = 1
    z += 1

#print(data_dic)
lis_f = sorted(data_dic, key=data_dic.get, reverse=True)
count = 0
draw_word_dic = {}
#print(lis_f)
while count < 10:
    draw_word_dic[str(lis_f[count])] = data_dic[str(lis_f[count])]
    count += 1
    

plt.barh(range(len(draw_word_dic)),draw_word_dic.values(),align='center')
plt.yticks(range(len(draw_word_dic)),list(draw_word_dic.keys()))
plt.xlabel('\nNumber of Papers')
plt.title('Trend of research in 2016 "SCOPUS" journal')
plt.ylabel('---- Areas ---- \n')
plt.show()

import sqlite3
import matplotlib.pyplot as plt
import operator

data = []
sqlite_database = '/home/neel/scopus_data/scopus_data.sqlite'
conn = sqlite3.connect(sqlite_database)
c = conn.cursor()
c.execute("SELECT `Year`,`IndexKeywords` FROM `AI_scopus` DESC LIMIT 0, 5000;")
data = c.fetchall()
#text = str(data[0])
#print(text[2:6])
#print(text[10:len(text)-2])
#tr = []
#tr = (text[10:len(text)-2]).split(";")
#print(tr)
conn.close()
data_dic = {}

z = 0 
word_lis = []
while z < len(data):
    text = str(data[z])
    year = str(text[2:6])
    #print(year)
    lis_word = (text[10:len(text)-2].replace(" ","")).split(";")
    #print(lis_word)
    if year == '2015':
        for word in lis_word:
            try:
                data_dic[str(word)] = int(data_dic[str(word)]) + 1
            except:
                data_dic[str(word)] = 1
    z += 1

#print(data_dic)
lis_f = sorted(data_dic, key=data_dic.get, reverse=True)
count = 0
draw_word_dic = {}
#print(lis_f)
while count < 10:
    draw_word_dic[str(lis_f[count])] = data_dic[str(lis_f[count])]
    count += 1
    

plt.barh(range(len(draw_word_dic)),draw_word_dic.values(),align='center')
plt.yticks(range(len(draw_word_dic)),list(draw_word_dic.keys()))
plt.xlabel('\nNumber of Papers')
plt.title('Trend of research in 2015 "SCOPUS" journal')
plt.ylabel('---- Areas ---- \n')
plt.show()

import sqlite3
import matplotlib.pyplot as plt
import operator

data = []
sqlite_database = '/home/neel/scopus_data/scopus_data.sqlite'
conn = sqlite3.connect(sqlite_database)
c = conn.cursor()
c.execute("SELECT `Year`,`IndexKeywords` FROM `AI_scopus` DESC LIMIT 0, 5000;")
data = c.fetchall()
#text = str(data[0])
#print(text[2:6])
#print(text[10:len(text)-2])
#tr = []
#tr = (text[10:len(text)-2]).split(";")
#print(tr)
conn.close()
data_dic = {}

z = 0 
word_lis = []
while z < len(data):
    text = str(data[z])
    year = str(text[2:6])
    #print(year)
    lis_word = (text[10:len(text)-2].replace(" ","")).split(";")
    #print(lis_word)
    if year == '2014':
        for word in lis_word:
            try:
                data_dic[str(word)] = int(data_dic[str(word)]) + 1
            except:
                data_dic[str(word)] = 1
    z += 1

#print(data_dic)
lis_f = sorted(data_dic, key=data_dic.get, reverse=True)
count = 0
draw_word_dic = {}
#print(lis_f)
while count < 10:
    draw_word_dic[str(lis_f[count])] = data_dic[str(lis_f[count])]
    count += 1
    

plt.barh(range(len(draw_word_dic)),draw_word_dic.values(),align='center')
plt.yticks(range(len(draw_word_dic)),list(draw_word_dic.keys()))
plt.xlabel('\nNumber of Papers')
plt.title('Trend of research in 2014 "SCOPUS" journal')
plt.ylabel('---- Areas ---- \n')
plt.show()

import sqlite3
import matplotlib.pyplot as plt
import operator

data = []
sqlite_database = '/home/neel/scopus_data/scopus_data.sqlite'
conn = sqlite3.connect(sqlite_database)
c = conn.cursor()
c.execute("SELECT `Year`,`IndexKeywords` FROM `AI_scopus` DESC LIMIT 0, 5000;")
data = c.fetchall()
#text = str(data[0])
#print(text[2:6])
#print(text[10:len(text)-2])
#tr = []
#tr = (text[10:len(text)-2]).split(";")
#print(tr)
conn.close()
data_dic = {}

z = 0 
word_lis = []
while z < len(data):
    text = str(data[z])
    year = str(text[2:6])
    #print(year)
    lis_word = (text[10:len(text)-2].replace(" ","")).split(";")
    #print(lis_word)
    if year == '2013':
        for word in lis_word:
            try:
                data_dic[str(word)] = int(data_dic[str(word)]) + 1
            except:
                data_dic[str(word)] = 1
    z += 1

#print(data_dic)
lis_f = sorted(data_dic, key=data_dic.get, reverse=True)
count = 0
draw_word_dic = {}
#print(lis_f)
while count < 10:
    draw_word_dic[str(lis_f[count])] = data_dic[str(lis_f[count])]
    count += 1
    

plt.barh(range(len(draw_word_dic)),draw_word_dic.values(),align='center')
plt.yticks(range(len(draw_word_dic)),list(draw_word_dic.keys()))
plt.xlabel('\nNumber of Papers')
plt.title('Trend of research in 2013 "SCOPUS" journal')
plt.ylabel('---- Areas ---- \n')
plt.show()

import sqlite3
import matplotlib.pyplot as plt
import operator

data = []
sqlite_database = '/home/neel/scopus_data/scopus_data.sqlite'
conn = sqlite3.connect(sqlite_database)
c = conn.cursor()
c.execute("SELECT `Year`,`IndexKeywords` FROM `AI_scopus` DESC LIMIT 0, 5000;")
data = c.fetchall()
#text = str(data[0])
#print(text[2:6])
#print(text[10:len(text)-2])
#tr = []
#tr = (text[10:len(text)-2]).split(";")
#print(tr)
conn.close()
data_dic = {}

z = 0 
word_lis = []
while z < len(data):
    text = str(data[z])
    year = str(text[2:6])
    #print(year)
    lis_word = (text[10:len(text)-2].replace(" ","")).split(";")
    #print(lis_word)
    if year == '2012':
        for word in lis_word:
            try:
                data_dic[str(word)] = int(data_dic[str(word)]) + 1
            except:
                data_dic[str(word)] = 1
    z += 1

#print(data_dic)
lis_f = sorted(data_dic, key=data_dic.get, reverse=True)
count = 0
draw_word_dic = {}
#print(lis_f)
while count < 10:
    draw_word_dic[str(lis_f[count])] = data_dic[str(lis_f[count])]
    count += 1
    

plt.barh(range(len(draw_word_dic)),draw_word_dic.values(),align='center')
plt.yticks(range(len(draw_word_dic)),list(draw_word_dic.keys()))
plt.xlabel('\nNumber of Papers')
plt.title('Trend of research in 2012 "SCOPUS" journal')
plt.ylabel('---- Areas ---- \n')
plt.show()

import sqlite3
import matplotlib.pyplot as plt
import operator

data = []
sqlite_database = '/home/neel/scopus_data/scopus_data.sqlite'
conn = sqlite3.connect(sqlite_database)
c = conn.cursor()
c.execute("SELECT `Year`,`IndexKeywords` FROM `AI_scopus` DESC LIMIT 0, 5000;")
data = c.fetchall()
#text = str(data[0])
#print(text[2:6])
#print(text[10:len(text)-2])
#tr = []
#tr = (text[10:len(text)-2]).split(";")
#print(tr)
conn.close()
data_dic = {}

z = 0 
word_lis = []
while z < len(data):
    text = str(data[z])
    year = str(text[2:6])
    #print(year)
    lis_word = (text[10:len(text)-2].replace(" ","")).split(";")
    #print(lis_word)
    if year == '2011':
        for word in lis_word:
            try:
                data_dic[str(word)] = int(data_dic[str(word)]) + 1
            except:
                data_dic[str(word)] = 1
    z += 1

#print(data_dic)
lis_f = sorted(data_dic, key=data_dic.get, reverse=True)
count = 0
draw_word_dic = {}
#print(lis_f)
while count < 10:
    draw_word_dic[str(lis_f[count])] = data_dic[str(lis_f[count])]
    count += 1
    

plt.barh(range(len(draw_word_dic)),draw_word_dic.values(),align='center')
plt.yticks(range(len(draw_word_dic)),list(draw_word_dic.keys()))
plt.xlabel('\nNumber of Papers')
plt.title('Trend of research in 2011 "SCOPUS" journal')
plt.ylabel('---- Areas ---- \n')
plt.show()

import sqlite3
import matplotlib.pyplot as plt
import operator

data = []
sqlite_database = '/home/neel/scopus_data/scopus_data.sqlite'
conn = sqlite3.connect(sqlite_database)
c = conn.cursor()
c.execute("SELECT `Year`,`IndexKeywords` FROM `AI_scopus` DESC LIMIT 0, 5000;")
data = c.fetchall()
#text = str(data[0])
#print(text[2:6])
#print(text[10:len(text)-2])
#tr = []
#tr = (text[10:len(text)-2]).split(";")
#print(tr)
conn.close()
data_dic = {}

z = 0 
word_lis = []
while z < len(data):
    text = str(data[z])
    year = str(text[2:6])
    #print(year)
    lis_word = (text[10:len(text)-2].replace(" ","")).split(";")
    #print(lis_word)
    if year == '2010':
        for word in lis_word:
            try:
                data_dic[str(word)] = int(data_dic[str(word)]) + 1
            except:
                data_dic[str(word)] = 1
    z += 1

#print(data_dic)
lis_f = sorted(data_dic, key=data_dic.get, reverse=True)
count = 0
draw_word_dic = {}
#print(lis_f)
while count < 10:
    draw_word_dic[str(lis_f[count])] = data_dic[str(lis_f[count])]
    count += 1
    

plt.barh(range(len(draw_word_dic)),draw_word_dic.values(),align='center')
plt.yticks(range(len(draw_word_dic)),list(draw_word_dic.keys()))
plt.xlabel('\nNumber of Papers')
plt.title('Trend of research in 2010 "SCOPUS" journal')
plt.ylabel('---- Areas ---- \n')
plt.show()

import sqlite3
import matplotlib.pyplot as plt

#fetching the name of different fields
name = []
#create the connection with database
sqlite_database = '/home/neel/scopus_data/scopus_data.sqlite'
conn = sqlite3.connect(sqlite_database)
c = conn.cursor()
c.execute("SELECT `name` FROM `university_data` ORDER BY `publish_paper` DESC LIMIT 0, 500;")
#store all name in as list
init_name = c.fetchall()
for each in init_name:
    text = (str(each)[2:len(each)-4]).replace("\\n","")
    name.append(text)
#close the connection with database
conn.close()

#fetching the number of publication field wise
sep = []
#connection create with database
sqlite_database = '/home/neel/scopus_data/scopus_data.sqlite'
conn = sqlite3.connect(sqlite_database)
c = conn.cursor()
c.execute("SELECT `publish_paper` FROM `university_data` ORDER BY `publish_paper` DESC LIMIT 0, 500;")
#store the data in sep as list
sep = c.fetchall()
#connection close with databae
conn.close()

#create a list of realtive percentage for publish paper field wise
per = []
for n in sep:
    text = str(n)[1:len(n)-3]
    n_to_per = int(text)
    val = (n_to_per*100)/1187
    val_2 = "%.2f"%val
    per.append(val_2)

#---------------------------Graph code------------------------------
label = []
x = 0
while x < len(per):
    label.append(str(name[x].upper())+" : "+str(per[x])+"%")
    x += 1

labels = label
sizes = per
patches, texts = plt.pie(sizes, startangle=90)
plt.legend(patches, labels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.title('Research done Top 15 Universitites and other Universities\n from 2001 to 2016\n Source: SCOPUS journal ')
plt.tight_layout()
plt.show()

import sqlite3
from matplotlib import pyplot as plt
from matplotlib import style
import matplotlib.pyplot as plt; plt.rcdefaults()

data = []
sqlite_database = '/home/neel/scopus_data/scopus_data.sqlite'
conn = sqlite3.connect(sqlite_database)
c = conn.cursor()
c.execute("SELECT `Year`,`IndexKeywords` FROM `AI_scopus` DESC LIMIT 0, 5000;")
data = c.fetchall()
text = str(data[0])
#print(text[2:6])
#print(text[10:len(text)-2])
#tr = []
tr = ((text[10:len(text)-2]).replace(" ","")).split(";")
#print(tr)
conn.close()
tred_word_dic = {}

data_ai = {}
data_nm = {}
data_ls = {}
data_algo = {}
data_cv = {}

field_lis = []

for line in data:
    text = str(line)
    year = text[2:6]
    field_lis = ((text[10:len(text)-2]).replace(" ","")).split(";")
    for field in field_lis:
        if field == 'Artificialintelligence':
            try:
                data_ai[year] = int(data_ai[year]) + 1
            except:
                data_ai[year] = 1
        if field == 'Neuralnetworks':
            try:
                data_nm[year] = int(data_nm[year]) + 1
            except:
                data_nm[year] = 1
        if field == 'Learningsystems':
            try:
                data_ls[year] = int(data_ls[year]) + 1
            except:
                data_ls[year] = 1
        if field == 'Algorithms':
            try:
                data_algo[year] = int(data_algo[year]) + 1
            except:
                data_algo[year] = 1
        if field == 'Computervision':
            try:
                data_cv[year] = int(data_cv[year]) + 1
            except:
                data_cv[year] = 1

x_xix = []
y_ai = []
y_nm = []
y_ls = []
y_algo = []
y_cv = []

x = 2001
zero = 0
while x < 2017:
    try:
        #print(x)
        y_ai.append(data_ai[str(x)])
        #print(data_CV[x])
    except:
        y_ai.append(int(zero))
        pass
    try:
        #print(x)
        y_nm.append(data_nm[str(x)])
        #print(data_CV[x])
    except:
        y_nm.append(int(zero))
        pass
    try:
        #print(x)
        y_ls.append(data_ls[str(x)])
        #print(data_CV[x])
    except:
        y_ls.append(int(zero))
        pass
    try:
        #print(x)
        y_algo.append(data_algo[str(x)])
        #print(data_CV[x])
    except:
        y_algo.append(int(zero))
        pass
    try:
        #print(x)
        y_cv.append(data_cv[str(x)])
        #print(data_CV[x])
    except:
        y_cv.append(int(zero))
        pass
    x_xix.append(x)
    x += 1
    
style.use('ggplot')
plt.plot(x_xix,y_cv,label="Computer Vision")
plt.plot(x_xix,y_ai,label="Artificial Intelligence")
plt.plot(x_xix,y_algo,label="Algorithms")
plt.plot(x_xix,y_ls,label="Learning Systems")
plt.plot(x_xix,y_nm,label="Neural Networks")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Trend of research in different realm of CS\n from 2001 to 2016')
plt.ylabel('Number of publish paper')
plt.xlabel('\nYears: 2001 - 2016')

plt.show()

import sqlite3
import matplotlib.pyplot as plt

#fetching the name of different fields
name = []
#create the connection with database
sqlite_database = '/home/neel/scopus_data/scopus_data.sqlite'
conn = sqlite3.connect(sqlite_database)
c = conn.cursor()
c.execute("SELECT `name` FROM `industry_data` ORDER BY `publish_paper` DESC LIMIT 0, 5000;")
#store all name in as list
init_name = c.fetchall()
for each in init_name:
    text = (str(each)[2:len(each)-4]).replace("\\n","")
    name.append(text)
#close the connection with database
conn.close()

#fetching the number of publication field wise
sep = []
#connection create with database
sqlite_database = '/home/neel/scopus_data/scopus_data.sqlite'
conn = sqlite3.connect(sqlite_database)
c = conn.cursor()
c.execute("SELECT `publish_paper` FROM `industry_data` ORDER BY `publish_paper` DESC LIMIT 0, 5000;")
#store the data in sep as list
sep = c.fetchall()
#connection close with databae
conn.close()

#create a list of realtive percentage for publish paper field wise
per = []
for n in sep:
    text = str(n)[1:len(n)-3]
    n_to_per = int(text)
    val = (n_to_per*100)/200
    val_2 = "%.2f"%val
    per.append(val_2)

#---------------------------Graph code------------------------------
label = []
x = 0
while x < len(per):
    label.append(str(name[x].upper())+" : "+str(per[x])+"%")
    x += 1

labels = label
sizes = per
patches, texts = plt.pie(sizes, startangle=90)
plt.legend(patches, labels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.title('Research percentage of different Industries\n from 2001 to 2016\n Source: SCOPUS journal ')
plt.tight_layout()
plt.show()

import sqlite3
import matplotlib.pyplot as plt

#fetching the name of different fields
name = []
#create the connection with database
sqlite_database = '/home/neel/scopus_data/scopus_data.sqlite'
conn = sqlite3.connect(sqlite_database)
c = conn.cursor()
c.execute("SELECT `name` FROM `seprate` DESC LIMIT 0, 5000;")
#store all name in as list
init_name = c.fetchall()
for each in init_name:
    text = (str(each)[2:len(each)-4]).replace("\\n","")
    name.append(text)
#close the connection with database
conn.close()

#fetching the number of publication field wise
sep = []
#connection create with database
sqlite_database = '/home/neel/scopus_data/scopus_data.sqlite'
conn = sqlite3.connect(sqlite_database)
c = conn.cursor()
c.execute("SELECT `number` FROM `seprate` DESC LIMIT 0, 5000;")
#store the data in sep as list
sep = c.fetchall()
#connection close with databae
conn.close()

#create a list of realtive percentage for publish paper field wise
per = []
for n in sep:
    text = str(n)[1:len(n)-3]
    n_to_per = int(text)
    val = (n_to_per*100)/1387
    val_2 = "%.2f"%val
    per.append(val_2)

#---------------------------Graph code------------------------------
label = []
x = 0
while x < len(per):
    label.append(str(name[x].upper())+" : "+str(per[x])+"%")
    x += 1

labels = label
sizes = per
patches, texts = plt.pie(sizes, startangle=90)
plt.legend(patches, labels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.title('Research done by Universities and Industries\n from 2001 to 2016\n Source: SCOPUS journal ')
plt.tight_layout()
plt.show()

