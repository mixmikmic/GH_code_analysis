get_ipython().run_line_magic('pylab', 'inline')
import warnings
warnings.filterwarnings('ignore')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Displaying max-height of 8000 px without scrolling, for cleaner visual representation
# source : stackoverflow
# https://stackoverflow.com/questions/18770504/resize-ipython-notebook-output-window

get_ipython().run_cell_magic('html', '', '<style>\n.output_wrapper, .output {\n    height:auto !important;\n    max-height:2000px;  /* your desired max-height here */\n}\n.output_scroll {\n    box-shadow:none !important;\n    webkit-box-shadow:none !important;\n}\n</style>')

get_ipython().run_cell_magic('time', '', "\n# Loading libraries\n\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport nltk\nfrom nltk.corpus import stopwords\n# nltk.download('stopwords')\n\n# Dependency: pip install tqdm\nfrom tqdm import tqdm_notebook as tqdm\n\nfrom collections import Counter")

# Loading data-set : reviews dataset with reviews of length 100-200 for restaurants ONLY.

get_ipython().run_cell_magic('time', '', "\n# col_names = ['review_id', 'business_id', 'user_id', 'text', 'stars', 'text length']\n\nreviews_dataset = pd.read_csv('reviews_restaurants_text.csv', low_memory= False)\ndisplay(reviews_dataset.head(3))")

# reviews_dataset.shape[0] - gives number of row count
print("Total No. of Reviews: {}".format(reviews_dataset.shape[0]))

reviews_dataset.shape

import string
def get_clean_text(sample_review):
    
    '''
    Takes in a string of text, then performs the following:
    1. Performs case normalization
    2. Remove all punctuation
    3. Remove all stopwords
    4. Return the cleaned text as a list of words
    '''
    stopwords = nltk.corpus.stopwords.words('english')
    newStopWords = ['ive','hadnt','couldnt','didnt', 'id']  ## more can also be added upon analysis
    stopwords.extend(newStopWords)
    text = sample_review
    #display(text)
    
    # text format of b'Review_starts' is beacuse of some encofing stuff, so we will remove it to make our review a 
    # string like 'sample review'
    text = text[2: len(sample_review)-1].lower()  ##  case normalization
    #display(text)
    
    text = text.replace('\\n', ' ').replace('\\t', ' ')
    #display(text)
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    #display(nopunc)
    
    l = [word for word in nopunc.split() if word.lower() not in stopwords]
    clean_text = ""
    for word in l:
        clean_text += str(word)+" "
    
    return clean_text.strip()

# TEST 
# get_clean_text()
sample_review = reviews_dataset.text[28]
display(get_clean_text(sample_review))

import string
def get_words(text):
    
    '''
    Takes in a string of text, then performs the following:
    1. Performs case normalization
    2. Remove all punctuation
    3. Remove all stopwords
    4. Return the cleaned text as a list of words
    '''
    stopwords = nltk.corpus.stopwords.words('english')
    newStopWords = ['ive','hadnt','couldnt','didnt', 'id']  ## more can also be added upon analysis
    stopwords.extend(newStopWords)
    
    
    # text format of b'Review_starts' is beacuse of some encofing stuff, so we will remove it to make our review a 
    # string like 'sample review'
    text = text[2: len(sample_review)-1].lower()  ##  case normalization
    #display(text)
    
    text = text.replace('\\n', ' ').replace('\\t', ' ')
    display(text)
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    display(nopunc)
    
    l = [word for word in nopunc.split() if word.lower() not in stopwords]
    
    return l, len(l)

for i in range(1):
    sample_review = str(reviews_dataset.text[i])
    #display(sample_review)
    check = get_words(sample_review)
    display(check[0]) # a tuple

pd.set_option('display.precision', 2)
reviews_dataset.describe()

reviews_dataset["stars"].value_counts()
type(reviews_dataset["stars"].value_counts())

labels = '5-Stars', '4-Stars', '1-Star', '3-Stars', '2-Stars'
sizes = reviews_dataset["stars"].value_counts()
colors = ['lightpink', 'yellowgreen', 'orange', 'lightskyblue','green']
 
# Plot
plt.pie(sizes, labels=labels, colors =colors, autopct='%1.1f%%') 
plt.axis('equal')
plt.show()

# Dataset is imbalanced, but it is taken care of by under-sampling when we are using the reviews dataset for recommendations

get_ipython().run_cell_magic('time', '', "# we're interested in the text of each review \n# and the stars rating, so we load these into \n# separate lists\n\ntexts = []\nstars = [reviews_dataset['stars'] for review in reviews_dataset]\npbar = tqdm(total=reviews_dataset.shape[0]+1)\nfor index, row in reviews_dataset.iterrows():\n    texts.append(get_clean_text(row['text']))\n    pbar.update(1)\npbar.close()")

# Vectorizing our Text Data - the TF-IDF algorithm along with n-grams
# and tokenization (splitting the text into individual words).

get_ipython().run_cell_magic('time', '', "# Estimated time: 29.8 s\nfrom sklearn.feature_extraction.text import TfidfVectorizer\n\n# This vectorizer breaks text into single words and bi-grams\n# and then calculates the TF-IDF representation\nvectorizer = TfidfVectorizer(ngram_range=(1,3))\n\n# the 'fit' builds up the vocabulary from all the reviews\n# while the 'transform' step turns each indivdual text into\n# a matrix of numbers.\nvectors = vectorizer.fit_transform(texts)")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(vectors, stars[1], test_size=0.15, random_state=42, shuffle =False)

# We now have 15% of our data in X_test and y_test. We’ll teach our system using 85%
# of the data (X_train and y_train), and then see how well it does by comparing its predictions for 
# the reviews in X_test with the real ratings in y_test.

get_ipython().run_cell_magic('time', '', '# Estimated time: 12.6 s\nfrom sklearn.svm import LinearSVC\n\n# initialise the SVM classifier\nclassifier = LinearSVC()\n\n# train the classifier\nclassifier.fit(X_train, y_train)')

# classifier has been fitted, it can now be used to make predictions. 
# predicting the rating for the first ten reviews in our test set

# Using our trained classifier to predict the ratings from text

preds = classifier.predict(X_test)
print("Actual Ratings(Stars): ",end = "")
display(y_test[:5])
print("Predicted Ratings: ",end = "")
print(preds[:5])

# Predicting for entire dataset

X_null, X_full_test, y_null, y_full_test = train_test_split(vectors, stars[1], test_size=0.999995, random_state=42, shuffle = False)
predict_all = classifier.predict(X_full_test)

predicted_stars = list(predict_all)

print("Actual Ratings(Stars): ")
print(y_full_test[154730:154736])
print("\nPredicted Ratings: ",end = "")
print(predicted_stars[154730:154736])

# Making new CSVs from dataframe

print("\nOriginal Reviews (with user bias)")
display(reviews_dataset.tail(10))

print("\nUnbiased Reviews (with predicted rating using user's review text)")
unbiased_reviews_dataset = reviews_dataset

# dropping actual ratings(stars) by user
unbiased_reviews_dataset = unbiased_reviews_dataset.drop('stars', 1)

# adding the unbiased predicted rating
unbiased_reviews_dataset['stars'] = predicted_stars

display(unbiased_reviews_dataset.tail(10))

# write dataframe to csv
file_name = "reviews_restaurants_text_unbiased_svm.csv"
unbiased_reviews_dataset.to_csv(file_name, encoding='utf-8', index=False)

# testing unbiased rating by loading from new csv file

#new_reviews_dataset = pd.read_csv('reviews_restaurants_text_unbiased_svm.csv', low_memory= False)
#display(new_reviews_dataset.tail(10))

# simplest method for evaluating such a system is to see the percentage of the time it accurately predicts the desired answer. 
# This method is unsurprisingly called accuracy. We can calculate the accuracy of our system by comparing the predicted reviews 
# and the real reviews–when they are the same, our classifier predicted the review correctly.
# We sum up all of the correct answers and divide by the total number of reviews in our test set. 
# If this number is equal to 1, it means our classifier was spot on every time.

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, preds))

# Precision and Recall are better for evaluating rather than using just accuracy measure.

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print ('Precision: ' + str(precision_score(y_test, preds, average='weighted')))
print ('Recall: ' + str(recall_score(y_test, preds, average='weighted')))

from sklearn.metrics import classification_report
print(classification_report(y_test, preds))

## Helper function for plotting confusion metrics

# citation: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
import itertools  
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

from sklearn import metrics
names = ['1','2','3','4','5']

# Compute confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, preds)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

# making binary classes
sentiments = []
for star in stars[1]:
    if star <= 3:
        sentiments.append('n')
    if star > 3:
        sentiments.append('p')

print(len(sentiments))        
        
## to see the effect without including 3, we need to undersample

get_ipython().run_cell_magic('time', '', '\nX2_train, X2_test, y2_train, y2_test = train_test_split(vectors, sentiments, test_size=0.20, random_state=42)')

get_ipython().run_cell_magic('time', '', '\nclassifier2 = LinearSVC()\n# train the classifier\nclassifier2.fit(X2_train, y2_train)')

preds2 = classifier2.predict(X2_test)
print("Actual Class:    ",end = "")
print(y2_test[:10])
print("\nPredicted Class: ",end = "")
print(list(preds2[:10]))

print(accuracy_score(y2_test, preds2))

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print ('Precision: ' + str(precision_score(y2_test, preds2, average='weighted')))
print ('Recall: ' + str(recall_score(y2_test, preds2, average='weighted')))

print(classification_report(y2_test, preds2))

print(metrics.confusion_matrix(y2_test, preds2))

class_names = ['negative','positive']

# Compute confusion matrix
cnf_matrix = metrics.confusion_matrix(y2_test, preds2)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

