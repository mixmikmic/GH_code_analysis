from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

#run this only once
nltk.download('stopwords')

posts = open('../data/raw_forum_posts.dat', 'r').read()

soup = BeautifulSoup(posts, 'lxml')
postTxt = soup.findAll('text')  #all posts <text> 
postDocs = [x.text for x in postTxt]
postDocs.pop(0)
postDocs = [x.lower() for x in postDocs]

stopset = set(stopwords.words('english'))
stopset.update(['lt','p','/p','br','amp','quot','field','font','normal','span','0px','rgb','style','51', 
                'spacing','text','helvetica','size','family', 'space', 'arial', 'height', 'indent', 'letter'
                'line','none','sans','serif','transform','line','variant','weight','times', 'new','strong', 'video', 'title'
                'white','word','letter', 'roman','0pt','16','color','12','14','21', 'neue', 'apple', 'class',  ])

#Before!
postDocs[0]

vectorizer = TfidfVectorizer(stop_words=stopset,
                                 use_idf=True, ngram_range=(1, 3))
X = vectorizer.fit_transform(postDocs)

X[0]

#After
print(X[0])

X.shape

lsa = TruncatedSVD(n_components=27, n_iter=100)
lsa.fit(X)


#This is the first row for V
lsa.components_[0]

import sys
print (sys.version)

terms = vectorizer.get_feature_names()
for i, comp in enumerate(lsa.components_): 
    termsInComp = zip (terms,comp)
    sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=True) [:10]
    print("Concept %d:" % i )
    for term in sortedTerms:
        print(term[0])
    print (" ")

lsa.components_



