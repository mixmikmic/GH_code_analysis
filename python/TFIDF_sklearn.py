import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

data = pickle.load(open('诗句.dat', 'rb'))

X_train, X_test, y_train, y_test = train_test_split(
    data['X'], data['y'],
    test_size=0.2, random_state=0
)

vectorizer = TfidfVectorizer(analyzer='char', lowercase=False, min_df=5, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
print('TFIDF dim: {}'.format(len(vectorizer.get_feature_names())))

X_test_vec = vectorizer.transform(X_test)

print(X_train_vec.shape, X_test_vec.shape, y_train.shape, y_test.shape)

def fit(clf, name=None):
    clf.fit(X_train_vec.toarray(), y_train)
    pred_train = clf.predict(X_train_vec.toarray())
    if name is not None:
        print(name)
    print('train precision: {}'.format(precision_score(y_train, pred_train)))
    print('train recall: {}'.format(recall_score(y_train, pred_train)))
    print('train f1: {}'.format(f1_score(y_train, pred_train)))
    pred_test = clf.predict(X_test_vec.toarray())
    print('test precision: {}'.format(precision_score(y_test, pred_test)))
    print('test recall: {}'.format(recall_score(y_test, pred_test)))
    print('test f1: {}'.format(f1_score(y_test, pred_test)))

fit(LinearSVC(random_state=0), name='LinearSVC')

fit(RandomForestClassifier(random_state=0, n_jobs=-1), name='RandomForestClassifier')

fit(KNeighborsClassifier(), name='KNeighborsClassifier')

fit(GaussianNB(), name='GaussianNB')



