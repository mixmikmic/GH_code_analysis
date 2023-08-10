#!pip install PyPDF2
# FRÜHJAHRSSESSION 2016, 28. Febr. - 17. Mrz. 2016, 88 - 655

import PyPDF2
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.style.use('ggplot')
import dateutil.parser
import re
import time

get_ipython().magic('matplotlib inline')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
datestring = time.strftime("%m-%h-%d")
datestring

pdfFileObj = open('NR_5002_1603.pdf', 'rb')     #'rb' for read binary mode
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
pdfReader.numPages

get_page = 88  #500
whole_Text_FrSe = ''

for page in range(88, 650):  #88 #650
    pageObj = pdfReader.getPage(get_page) 
    Page = pageObj.extractText()
    get_page += 1
    whole_Text_FrSe = whole_Text_FrSe + Page

pdfFileObj = open('NR_5003_1604.pdf', 'rb')     #'rb' for read binary mode
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
pdfReader.numPages

get_page = 22
whole_Text_SonSe = ''

for page in range(22, 158):  #0 26
    pageObj = pdfReader.getPage(get_page) 
    Page = pageObj.extractText()
    get_page += 1
    whole_Text_SonSe = whole_Text_SonSe + Page

pdfFileObj = open('Bulletin_Sommersession_NR_5004_1606.pdf', 'rb')     #'rb' for read binary mode
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
pdfReader.numPages

get_page = 80
whole_Text_SomSe = ''

for page in range(80, 587):  #0 26
    pageObj = pdfReader.getPage(get_page) 
    Page = pageObj.extractText()
    get_page += 1
    whole_Text_SomSe = whole_Text_SomSe + Page

whole_Text = whole_Text_FrSe + whole_Text_SonSe + whole_Text_SomSe

#Removing Zwischenrufe der Präsidentin
whole_Text = whole_Text.replace('(Zwischenruf', "") 
#Getting rid of \n, makes search through the document more difficult, because I can't use "." to find different characters
whole_Text = whole_Text.replace('\n', " ")
#Getting rid of double spaces
whole_Text = whole_Text.replace('  ', " ")
#Getting rid of '  -' spaces
whole_Text = whole_Text.replace('  -', " ")
#Getting rid of '  -' spaces
whole_Text = whole_Text.replace('-  ', " ")
#Getting rid of '- ' spaces
whole_Text = whole_Text.replace('- ', " ")
#Getting rid of ' -' spaces
whole_Text = whole_Text.replace(' -', " ")
#Formating –
whole_Text = whole_Text.replace('Œ', "–") 
#Headers and bottom of the page
#Bulletin officiel de l’Assemblée fédérale
whole_Text = whole_Text.replace('Bulletin officiel de l™Assemblée fédérale', " ")
#Bulletin officiel de l’Assemblée fédérale
whole_Text = whole_Text.replace('Amtliches Bulletin der Bundesversammlung', " ") 
#Removing confusing brackets
whole_Text = whole_Text.replace('(GE,JU, VS, TI)', '').replace('(GE, JU, VS, TI)','')
whole_Text = whole_Text.replace('(LU, SZ, GL, SO, BS, AI).', '')
#Removing page headers
whole_Text = whole_Text.replace('           Conseil national', '')
whole_Text = whole_Text.replace('Nationalrat           ', '')
#Deleting more brakets
whole_Text = re.sub(r'\([1-9]+\.[1-9]+\)', '', whole_Text) #(12.502)
whole_Text = re.sub(r'\(\s*[A-Z][a-z][A-Za-z]+\s*\)', '', whole_Text) #(KdK) #(Munz) #(Walter)
whole_Text = re.sub(r'\(\s*[A-Z][A-Z][A-Z]+\s*\)', '', whole_Text) #(KKK ) 
whole_Text = re.sub(r'\([A-Z[a-z]+\s*[A-Z[a-z]+\)', '', whole_Text) #(Erich Hess)
whole_Text = whole_Text.replace('\(Aktienrecht\)', '') #(Aktienrecht)

#Finding everthing in brackets, i.e. '(GL, AG)'
Partei_Kanton = re.findall(r"\([A-Z]+, [A-Z][A-Z]\)", whole_Text)
len(Partei_Kanton)

#Get name from before the pattern (XY, XY), using the negavitve outlook:
#http://stackoverflow.com/questions/31713623/search-in-a-string-and-obtain-the-2-words-before-and-after-the-match-in-python
Personen_name = re.findall(r'\w*\s*\w+\b\s+\b[-\w|\w]+(?=\s*\([A-Z]+, [A-Z][A-Z]\))', whole_Text)
len(Personen_name)

#Getting the whole text
Text = re.findall(r"[A-Z],\s+[A-Z][A-Z]\)(.*?)(Leuthard Doris|Sommaruga Simonetta|Schneider-Ammann Johann|Maurer Ueli|Burkhalter Didier|Berset Alain|Parmelin Guy|\w*\.Leuthard Doris|\w*\.Sommaruga Simonetta|\w*\.Schneider-Ammann Johann|\w*\.Maurer Ueli|\w*\.Burkhalter Didier|\w*\.Berset Alain|\w*\.Parmelin Guy|\()", whole_Text)
#Removing empty lists
Text = list(filter(None, Text))
#BEcause it also captures the Bundesrats name, when I don't want it two, I have unpack the lists
Text = ([ a for a,b in Text ], [ b for a,b in Text ])
#And then only take the first list
Text = Text[0]
len(Text)

#for testing, because in the test the last text will never be included
#Partei_Kanton.pop()

#for testing, because in the test the last text will never be included
#Personen_name.pop()

def delete_Bundesrat(List):
    for x in List:
        x = re.sub(r'Simonetta Sommaruga,.*?$', '', x)
        x = re.sub(r'Maurer Ueli,.*?$', '', x) 
        x = re.sub(r'Leuthard Doris,.*?$', '', x) 
        x = re.sub(r'Burkhalter Didier,.*?$', '', x)
        x = re.sub(r'Berset Alain,.*?$', '', x)
        x = re.sub(r'Parmelin Guy,.*?$', '', x)
        return x

delete_Bundesrat(Text)

def delete_Bundesrat2(List):
    for x in List:
        return x.replace('Bei diesem Block 2', '')

delete_Bundesrat2(Text)

#Starting Dictionary
Dict_List = []
Dict_Text = []
for Partei_Kanton, Name, Talk in zip(Partei_Kanton, Personen_name, Text):
        
        #Getting Partei
        Partei = re.search(r'[A-Z]+,', Partei_Kanton)
        Partei = Partei.group().replace(',', '')
    
        #Getting Kanton
        Kanton = re.search(r', [A-Z][A-Z]', Partei_Kanton)
        Kanton = Kanton.group().replace(', ', '')
        
        #Getting Name
        Name = Name.replace('  ', ' ').strip()
        Name = Name.replace('16 ', '')
        Name = re.sub(r'[0-9]*', '', Name).strip()
        
        #Sometimes the Regex grabs words is should, so I am filtering them out here. Not ideal though.
        Name = Name.replace('motion', '').replace('fédérale', '').replace('Etats', '').replace('postulat', '').replace('initiative', '').replace('Maintenir', '').replace('sports', '').replace('finances', '').replace('Biffer', '').strip()
        Name = Name.replace('matière ', '').replace('renvoi', '').replace('intérieur ', '').strip()
        
        #I still need to get rid of name at the end of each talk:
        #Talk
        
        #Getting lenght of talk
        Talk_len = len(Talk)
    
        Dict = {'Partei': Partei,
               'Kanton': Kanton,
               'Name': Name,
                'Reden_Länge': Talk_len
               }
        
        Text_Book = {'Partei': Partei,
               'Kanton': Kanton,
               'Name': Name,
                'Reden': Talk,
                'Reden_Länge': Talk_len
               }
        Dict_List.append(Dict)
        Dict_Text.append(Text_Book)

#Making the DataFrame for politicians & 
df = pd.DataFrame(Dict_List)

#Making DataFrame to look up texts
df_Text = pd.DataFrame(Dict_Text)

df.head()

#What is the average reading speed per minute. Slide presentations are around
#120 words per minute, but these are policy speakers, who speak faster. 
#we'll word with the average: 200 words per minute. https://en.wikipedia.org/wiki/Words_per_minute
#Average word length in German is 5 - 6. We'll work with 5.5.

df['Rede Minuten'] = df['Reden_Länge'] / 1100

#Making the plot look nice:
#http://jonathansoma.com/lede/data-studio/classes/tufte/matplotlib-styles-data-ink-and-annotation/

# Making default font 
matplotlib.rcParams['font.sans-serif'] = "DIN Condensed"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "DIN Condensed"

fig, ax = plt.subplots(figsize =(5,5), facecolor='WhiteSmoke')

df_TOTAL_MINs = df.groupby('Partei')['Rede Minuten'].sum().sort_values(ascending=False).plot(kind='pie', colors=['LawnGreen', 'tomato', 'b', 'orange', 'SeaGreen', 'YellowGreen', 'Gold'], ax=ax)
ax.set_ylabel("")

#Setting title
ax.set_title("Aufteilung der Rede-Minuten im Nationalrat", fontname='DIN Condensed', fontsize=24)



df.groupby('Partei')['Rede Minuten'].sum().sort_values

#This looks as though it corresponds pretty well to the percentages of
#the various parties in parliament. Lets have a closer look though. What is the 
#Total number of minutes?

#creating dataframe
df_TOTAL_MINs = df.groupby('Partei')['Rede Minuten'].sum()
df_TOTAL_MINs = pd.DataFrame(df_TOTAL_MINs)
#Re-Indexing
df_TOTAL_MINs.reset_index(inplace=True)
#Creating Anteil
df_TOTAL_MINs['Anteil'] = df_TOTAL_MINs['Rede Minuten'] / df['Rede Minuten'].sum() * 100

#Comparing to atual size of the party. Franktionen here:
#https://www.parlament.ch/de/organe/fraktionen
#Check this against the speeches
df_NR_FRAKTIONEN = pd.read_csv("FRAKTIONEN_NR.csv")
df_TOTAL_MINs = df_TOTAL_MINs.merge(df_NR_FRAKTIONEN, left_on = 'Partei', right_on ='Partei') 
#Renaming the columns
df_TOTAL_MINs.columns = ['Partei', 'Rede Minuten', 'Anteil Reden', 'Anzahl Mitglieder', 'Anteil im Parlament']
#Working out the percentage gained or lost
df_TOTAL_MINs['Gewinn_Verlust'] = df_TOTAL_MINs['Anteil Reden'] - df_TOTAL_MINs['Anteil im Parlament']
#Working out the minutes gaines or lost
df_TOTAL_MINs['In Minuten'] = df_TOTAL_MINs['Gewinn_Verlust'] * (df['Rede Minuten'].sum() / 100)

# Anteile im Parlament
# Making default font 
matplotlib.rcParams['font.sans-serif'] = "DIN Condensed"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "DIN Condensed"

fig, ax = plt.subplots(figsize =(5,5), facecolor='WhiteSmoke')

df_TOTAL_MINs['Anteil im Parlament'].plot(kind='pie', colors=['Gold', 'orange', 'SeaGreen', 'YellowGreen', 'b', 'tomato',  'LawnGreen'], ax=ax)
ax.set_ylabel("")

#Setting title
ax.set_title("Aufteile der Fraktionen im Nationalrat", fontname='DIN Condensed', fontsize=24)

df_TOTAL_MINs['Anteil im Parlament']

df_TOTAL_MINs

#Gewinner und Verlierer 

fig, ax = plt.subplots(figsize =(5,5), facecolor='White')
df_TOTAL_MINs['In Minuten'].plot(kind='bar', color=['Gold', 'orange', 'SeaGreen', 'YellowGreen', 'b', 'tomato',  'LawnGreen'])

ax.set_title("Die Grünen und Grünliberalen im Plus", fontname='DIN Condensed', fontsize=24)

ax.set_ylabel(ylabel='')
ax.set_xlabel(xlabel='')
ax.set_axis_bgcolor("White")

plt.tick_params(
    #axis='x',
    top='off',
    which='major',
    left='off',
    right='on',
    bottom='off',
    labeltop='off',
    labelbottom='on',
    labelright='on',
    labelleft='off')

ax.set_axisbelow(True)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.set_title("Debattier-Muffel SVP", fontname='DIN Condensed', fontsize=24)
plt.savefig('2.pdf', transparent=True, bbox_inches='tight')

df_TOTAL_MINs['In Minuten']

fig, ax = plt.subplots(figsize =(5,5), facecolor='White')
df_TOTAL_MINs['Rede Minuten'].plot(kind='bar', color=['Gold', 'orange', 'SeaGreen', 'YellowGreen', 'b', 'tomato',  'LawnGreen'])

ax.set_ylabel(ylabel='')
ax.set_xlabel(xlabel='')
ax.set_axis_bgcolor("White")

plt.tick_params(
    #axis='x',
    top='off',
    which='major',
    left='off',
    right='on',
    bottom='off',
    labeltop='off',
    labelbottom='on',
    labelright='on',
    labelleft='off')

ax.set_axisbelow(True)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.set_title("Die SP sticht am längsten", fontname='DIN Condensed', fontsize=24)
plt.savefig('1.pdf', transparent=True, bbox_inches='tight')



#First, was is the average speech length:
df['Rede Minuten'].mean()

df['Rede Minuten'].median()

fig, ax = plt.subplots(figsize =(10,5), facecolor='White')
df.groupby('Kanton')['Rede Minuten'].mean().sort_values(ascending=True).plot(kind='bar')
ax.set_ylabel(ylabel='')
ax.set_xlabel(xlabel='')
ax.set_axis_bgcolor("White")

plt.tick_params(
    #axis='x',
    top='off',
    which='major',
    left='off',
    right='on',
    bottom='off',
    labeltop='off',
    labelbottom='on',
    labelright='on',
    labelleft='off')

ax.set_axisbelow(True)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.set_title("Die Redelänge der Kantone", fontname='DIN Condensed', fontsize=24)
plt.savefig('6.pdf', transparent=True, bbox_inches='tight')

#lets try a scatter plot, marking Kantons from East to West, 1 - 26. First I need to create that file and import it. 

df_ostwest = pd.read_csv("reden_ostwest.csv")

df = df.merge(df_ostwest, left_on = 'Kanton', right_on ='Kanton') 

#UR is missing
fig, ax = plt.subplots(figsize =(10,5), facecolor='White')
df.plot(kind='scatter', x='Ost_West', y='Rede Minuten', ax=ax, marker='o')

#No Correlation
df.corr()

#Preparing the dataframe for sex of NR
df_NR_extended = pd.read_csv("National- und Ständeräte - NR.csv")

#Meging the Dataframe
df = df.merge(df_NR_extended, left_on = 'Name', right_on ='Name') 

df.groupby('Geschlecht')['Rede Minuten'].mean()

df.head(1)

df.groupby('Arbeitssprache')['Rede Minuten'].mean()

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(facecolor='WhiteSmoke', nrows=3, ncols=3, figsize=(6,6))

df[df['Partei'] == 'V'].groupby('Geschlecht')['Rede Minuten'].sum().plot(ax=ax1, kind='pie', startangle=270, colors=['b', 'r'])
ax1.set_title("SVP")
df[df['Partei'] == 'S'].groupby('Geschlecht')['Rede Minuten'].sum().plot(ax=ax2, kind='pie', startangle=270, colors=['b', 'r'])
ax2.set_title("SP")
df[df['Partei'] == 'RL'].groupby('Geschlecht')['Rede Minuten'].sum().plot(ax=ax3, kind='pie', startangle=270, colors=['b', 'r'])
ax3.set_title("FDP")
df[df['Partei'] == 'G'].groupby('Geschlecht')['Rede Minuten'].sum().plot(ax=ax4, kind='pie', startangle=270, colors=['b', 'r'])
ax4.set_title("Grüne")
df[df['Partei'] == 'GL'].groupby('Geschlecht')['Rede Minuten'].sum().plot(ax=ax5, kind='pie', startangle=270, colors=['b', 'r'])
ax5.set_title("Grünliberale")
df[df['Partei'] == 'BD'].groupby('Geschlecht')['Rede Minuten'].sum().plot(ax=ax6, kind='pie', startangle=270, colors=['b', 'r'])
ax6.set_title("BDP")
df[df['Partei'] == 'C'].groupby('Geschlecht')['Rede Minuten'].sum().plot(ax=ax7, kind='pie', startangle=270, colors=['b', 'r'])
ax7.set_title("CVP")

ax1.set_ylabel('')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax4.set_ylabel('')
ax5.set_ylabel('')
ax6.set_ylabel('')
ax7.set_ylabel('')


#ax1.set_title("Anteil Redelängen, Männer vs. Frauen", fontname='DIN Condensed', fontsize=24)
#plt.tight_layout()
plt.savefig('4.pdf', transparent=True, bbox_inches='tight')

#Comparing that to actual size of woman votes.

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(facecolor='WhiteSmoke', nrows=3, ncols=3, figsize=(6,6))

df_NR_extended[df_NR_extended['Fraktion'] == 'V']['Geschlecht'].value_counts().plot(ax=ax1, kind='pie', startangle=270, colors=['b', 'r'])
ax1.set_title("SVP")
df_NR_extended[df_NR_extended['Fraktion'] == 'S']['Geschlecht'].value_counts().plot(ax=ax2, kind='pie', startangle=270, colors=['b', 'r'])
ax2.set_title("SP")
df_NR_extended[df_NR_extended['Fraktion'] == 'RL']['Geschlecht'].value_counts().plot(ax=ax3, kind='pie', startangle=270, colors=['b', 'r'])
ax3.set_title("FDP")
df_NR_extended[df_NR_extended['Fraktion'] == 'G']['Geschlecht'].value_counts().plot(ax=ax4, kind='pie', startangle=270, colors=['b', 'r'])
ax4.set_title("Grüne")
df_NR_extended[df_NR_extended['Fraktion'] == 'GL']['Geschlecht'].value_counts().plot(ax=ax5, kind='pie', startangle=270, colors=['b', 'r'])
ax5.set_title("Grünliberale")
df_NR_extended[df_NR_extended['Fraktion'] == 'BD']['Geschlecht'].value_counts().plot(ax=ax6, kind='pie', startangle=270, colors=['b', 'r'])
ax6.set_title("BDP")
df_NR_extended[df_NR_extended['Fraktion'] == 'C']['Geschlecht'].value_counts().plot(ax=ax7, kind='pie', startangle=270, colors=['b', 'r'])
ax7.set_title("CVP")

ax1.set_ylabel('')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax4.set_ylabel('')
ax5.set_ylabel('')
ax6.set_ylabel('')
ax7.set_ylabel('')

plt.savefig('5.pdf', transparent=True, bbox_inches='tight')

df.groupby('Name')['Rede Minuten'].sum().sort_values(ascending=True).tail(10).plot(kind='barh')
plt.savefig('3.pdf', transparent=True, bbox_inches='tight')

df.groupby('Name')['Rede Minuten'].sum().sort_values(ascending=True).tail(10)

df.groupby('Name')['Rede Minuten'].sum().describe()

df.groupby('Name')['Reden_Länge'].sum().sort_values(ascending=False).head()

df.groupby('Name')['Rede Minuten'].sum().sort_values(ascending=False).tail(10).plot(kind='barh')

df_Text[df_Text['Name'] == 'Aeschi Thomas']  

df_Text[df_Text['Name'] == 'Schelbert Louis'].to_csv('schelbert_check.csv', index=False)
df_Text[df_Text['Name'] == 'Vogler Karl'].to_csv('vogler_check.csv', index=False)

df['Reden_Länge'].describe()

df['Rede Minuten'].sort_values(ascending=False)
df[df['Reden_Länge'] == 12888]

df_Text[df_Text['Name'] == 'Amaudruz Céline'].to_csv('amaudraz.csv', index=False)



df.plot(kind='scatter', x='VereidigungsDatum', y='Rede Minuten')

df.corr()

df_Skala = pd.read_csv("Skala4.csv")
df = df.merge(df_Skala, left_on = 'Partei', right_on ='Partei') 

df.plot(kind='scatter', y='Rede Minuten', x='Grösse')

SUM = pd.DataFrame(df.groupby('Name')['Rede Minuten'].sum())
SUM.reset_index(inplace=True)
SUM = SUM.merge(df, left_on = 'Name', right_on ='Name') 

SUM.plot(kind='scatter', x='Grösse', y='Rede Minuten_x')







