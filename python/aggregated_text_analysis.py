import re
punctuation = ['(', ')', '?', ':', ';', ',', '.', '!', '/', '"', "'"]
ratings_words_list = [ones, twos, threes, fours, fives]
cleaned_word_list = {}
for rating_word_list in ratings_words_list:
    rating_word_list = [x.split(".") for x in rating_word_list]
    rating_word_list = [item for sublist in rating_word_list for item in sublist]
    rating_word_list = [x.split(" ") for x in rating_word_list]
    rating_word_list = [item for sublist in rating_word_list for item in sublist]
    rating_word_list = [x for x in rating_word_list if re.match('[a-zA-Z]', x)]
    rating_word_list = [re.sub('\n','', x) for x in rating_word_list]
    rating_word_list = [x.lower() for x in rating_word_list]

#Build a test set and a training set for reviews
test_amount = int(0.8 * len(review_list))
train_reviews = review_list[0:test_amount] 
train_stars = star_list[0:test_amount]
test_reviews = review_list[test_amount:len(review_list)] 
test_stars = star_list[test_amount:len(star_list)] 

##Use the Harvard-IV negative Dictionary
vectorizer = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 vocabulary = cleaned_words,                              stop_words = 'english',                                max_features = None) 

tf_transformer = TfidfTransformer(use_idf=True)

train_data_features = vectorizer.fit_transform(train_reviews)
test_data_features = vectorizer.transform(test_reviews)

train_data_features = tf_transformer.fit_transform(train_data_features)
test_data_features = tf_transformer.fit_transform(test_data_features)
vocab = vectorizer.get_feature_names()

lin_svm = svm.LinearSVC(multi_class='crammer_singer')
lin_svm = lin_svm.fit(train_data_features, train_stars)
lin_svm_result = lin_svm.predict(test_data_features)

output = pd.DataFrame( data={"Reviews": test_reviews, "Rating": test_stars, "Predicted_Rating":lin_svm_result} )
output['Lin_SVM_Accurate'] = np.where(output['Rating'] == output['Predicted_Rating'], 1, 0)
accurate_percentage = float(sum(output['Lin_SVM_Accurate']))/float(len(output))

print accurate_percentage

#Try a RandomForest classifier
forest = RandomForestClassifier(n_estimators = 200, criterion ='entropy')
forest = forest.fit(train_data_features, train_stars)
rf_result = forest.predict(test_data_features)

output = pd.DataFrame( data={"Reviews": test_reviews, "Rating": test_stars, "Predicted_Rating":rf_result} )
output['RF_Accurate'] = np.where(output['Rating'] == output['Predicted_Rating'], 1, 0)
accurate_percentage = float(sum(output['RF_Accurate']))/float(len(output))

print accurate_percentage

bag_dt = BaggingClassifier(n_estimators=200, n_jobs=-1)
bag_dt = bag_dt.fit(train_data_features, train_stars)
bag_dt_result = bag_dt.predict(test_data_features)

output = pd.DataFrame( data={"Reviews": test_reviews, "Rating": test_stars, "Predicted_Rating":bag_dt_result} )
output['Bag_DT_Accurate'] = np.where(output['Rating'] == output['Predicted_Rating'], 1, 0)
accurate_percentage = float(sum(output['Bag_DT_Accurate']))/float(len(output))

print accurate_percentage

svm_classifier = SGDClassifier(loss='perceptron', shuffle = False, eta0=10e-100, learning_rate='invscaling')
svm_classifier = svm_classifier.fit(baseline_train_data_features, train_stars)
svm_result = svm_classifier.predict(baseline_test_data_features)

output = pd.DataFrame( data={"Reviews": test_reviews, "Rating": test_stars, "Predicted_Rating":svm_result} )
output['SVM_Accurate'] = np.where(output['Rating'] == output['Predicted_Rating'], 1, 0)
accurate_percentage = float(sum(output['SVM_Accurate']))/float(len(output))

print accurate_percentage

#Load in the Harvard-IV Sentiment Dictionary, then create a new list of only the negative words
harvard_dict = pd.read_csv('HIV-4.csv')
neg_words = list(set(harvard_dict[(harvard_dict.Negativ == 'Negativ')].Entry))
neg_words = [x.lower() for x in pos_neg_words]
cleaned_neg_words = []
for word in pos_neg_words:
    word = re.sub("#", "", word)
    word = re.sub("\d", "", word)
    cleaned_neg_words.append(word)
cleaned_neg_words = list(set(cleaned_words))

#Here we build a list of sentiment words using past studies and analysis
sentiment_words = ['worst', 'rude', 'terrible', 'horrible', 'bad', 'soggy', 'disappointing', 'overcooked', 'sorry',
                 'awful', 'disgusting', 'bland', 'tasteless', 'gross', 'mediocre', 'worse', 'poor', 
                 'sexy', 'sensual', 'seductive', 'voluptuously', 'ravishing', 'ok', 'perfection', 'thank',
                 'loved', 'reasonable', 'incredible', 'masterpiece', 'responsible', 'top-quality', 'fantastic!',
                 'incompetent', 'fuck-up', 'spiders', 'yikes', 'ant', 'overpriced',
                 'hedonistic', 'drug', 'addicting', 'addicted', 'sad', 'barely', 'favorite', 'favorites', 'die',
                 'awesome', 'glad', 'delicious', 'dry', 'money', 'unfortunately', 'frozen']

for n,i in enumerate(star_list):
    if i==1:
        star_list[n]='bad'
    if i==2:
        star_list[n]='bad'
    if i==3:
        star_list[n]='bad'
    if i==4:
        star_list[n]='good'
    if i==5:
        star_list[n]='good'

#Separate out a random set of 5000 reviews into different ratings lists, then find the unique words in each list
fives = []
fours = []
threes = []
twos = []
ones = []
for state in states:
    for review in state_dict[state][0:1000]:
        if review['stars'] == 5:
            fives.append(review['text'])
        elif review['stars'] == 4:
            fours.append(review['text'])
        elif review['stars'] == 3:
            threes.append(review['text'])
        elif review['stars'] == 2:
            twos.append(review['text'])
        elif review['stars'] == 1:
            ones.append(review['text'])

threes = [x.split(".") for x in threes]
threes = [item for sublist in threes for item in sublist]
threes = [x.split(" ") for x in threes]
threes = [item for sublist in threes for item in sublist]
threes = [x for x in threes if re.match('[a-zA-Z]', x)]
threes = [re.sub('\n','', x) for x in threes]
threes = [x.lower() for x in threes]

