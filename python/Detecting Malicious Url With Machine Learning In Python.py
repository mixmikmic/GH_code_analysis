# EDA Packages
import pandas as pd
import numpy as np
import random


# Machine Learning Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load Url Data 
urls_data = pd.read_csv("urldata.csv")

type(urls_data)

urls_data.head()

# Check for missing data
urls_data.isnull().sum().sum()

def makeTokens(f):
    tkns_BySlash = str(f.encode('utf-8')).split('/')	# make tokens after splitting by slash
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-')	# make tokens after splitting by dash
        tkns_ByDot = []
        for j in range(0,len(tokens)):
            temp_Tokens = str(tokens[j]).split('.')	# make tokens after splitting by dot
            tkns_ByDot = tkns_ByDot + temp_Tokens
        total_Tokens = total_Tokens + tokens + tkns_ByDot
    total_Tokens = list(set(total_Tokens))	#remove redundant tokens
    if 'com' in total_Tokens:
        total_Tokens.remove('com')	#removing .com since it occurs a lot of times and it should not be included in our features
    return total_Tokens

# Labels
y = urls_data["label"]

# Features
url_list = urls_data["url"]

# Using Default Tokenizer
#vectorizer = TfidfVectorizer()

# Using Custom Tokenizer
vectorizer = TfidfVectorizer(tokenizer=makeTokens)

# Store vectors into X variable as Our XFeatures
X = vectorizer.fit_transform(url_list)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)	

# Model Building
#using logistic regression
logit = LogisticRegression()	
logit.fit(X_train, y_train)

# Accuracy of Our Model Using Test Data
print("Accuracy ",logit.score(X_test, y_test))

# Accuracy of Our Model Using Train Data
print("Accuracy ",logit.score(X_train, y_train))

X_predict = ["google.com/search=jcharistech",
"google.com/search=faizanahmad",
"pakistanifacebookforever.com/getpassword.php/", 
"www.radsport-voggel.de/wp-admin/includes/log.exe", 
"ahrenhei.without-transfer.ru/nethost.exe ",
"www.itidea.it/centroesteticosothys/img/_notes/gum.exe"]

X_predict = vectorizer.transform(X_predict)
New_predict = logit.predict(X_predict)

print(New_predict)

# https://db.aa419.org/fakebankslist.php
X_predict1 = ["www.buyfakebillsonlinee.blogspot.com", 
"www.unitedairlineslogistics.com",
"www.stonehousedelivery.com",
"www.silkroadmeds-onlinepharmacy.com" ]

X_predict1 = vectorizer.transform(X_predict1)
New_predict1 = logit.predict(X_predict1)
print(New_predict1)

# Using Default Tokenizer
vectorizer = TfidfVectorizer()

# Store vectors into X variable as Our XFeatures
X = vectorizer.fit_transform(url_list)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)	

# Model Building

logitmodel = LogisticRegression()	#using logistic regression
logitmodel.fit(X_train, y_train)

# Accuracy of Our Model with our Custom Token
print("Accuracy ",logitmodel.score(X_test, y_test))

X_predict2 = ["www.buyfakebillsonlinee.blogspot.com", 
"www.unitedairlineslogistics.com",
"www.stonehousedelivery.com",
"www.silkroadmeds-onlinepharmacy.com" ]

X_predict2 = vectorizer.transform(X_predict2)
New_predict2 = logitmodel.predict(X_predict2)
print(New_predict2)

from sklearn.metrics import confusion_matrix

predicted = logitmodel.predict(X_test)
matrix = confusion_matrix(y_test, predicted)

print(matrix)



from sklearn.metrics import classification_report

report = classification_report(y_test, predicted)

print(report)

# Visualization Packages
import matplotlib.pyplot as plt
import seaborn as sns

matrix

plt.figure(figsize=(20,10))

# Confusion Matrix Graph With Seaborn
sns.heatmap(matrix,annot=True)
plt.show()

# Setting formate to integer with "d"
sns.heatmap(matrix,annot=True,fmt="d")
plt.show()

# Plot with Labels

plt.title('Confusion Matrix ')

sns.heatmap(matrix,annot=True,fmt="d")
# Set x-axis label
plt.xlabel('Predicted Class')
# Set y-axis label
plt.ylabel('Actual Class')
plt.show()

# Thanks For Watching
#J-Secur1ty
#Jesus Saves @ JCharisTech

