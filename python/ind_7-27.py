get_ipython().run_cell_magic('writefile', 'mornApp/build_model.py', 'import pandas as pd\nimport matplotlib.pyplot as plt\nimport numpy as np\n\ncategories = [\'alt.atheism\', \'soc.religion.christian\', \'comp.graphics\', \'sci.med\']\nfrom sklearn.datasets import fetch_20newsgroups\ntwenty_train = fetch_20newsgroups(subset=\'train\',categories=categories, shuffle=True, random_state=42)\n\nfrom sklearn.feature_extraction.text import CountVectorizer\ncount_vect = CountVectorizer()\nX_train_counts = count_vect.fit_transform(twenty_train.data)\n\nfrom sklearn.feature_extraction.text import TfidfTransformer\ntf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)\nX_train_tf = tf_transformer.transform(X_train_counts)\n\ntfidf_transformer = TfidfTransformer()\nX_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)\n\nfrom sklearn.naive_bayes import MultinomialNB\nclf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)\n\nimport pickle\npickle.dump(count_vect, open( "mornApp/data/my_vectorizer.pkl", "wb" ))\npickle.dump(clf, open( "mornApp/data/my_model.pkl", "wb" ))\npickle.dump(tfidf_transformer, open( "mornApp/data/my_transformer.pkl", "wb" ))\nclf2 = pickle.load(open( "mornApp/data/my_model.pkl", "rb" ) )\ncount_vect2 = pickle.load(open( "mornApp/data/my_vectorizer.pkl", "rb" ) )\n\ndocs_new = [\'God is love\', \'OpenGL on the GPU is fast\']\nX_new_counts = count_vect2.transform(docs_new)\nX_new_tfidf = tfidf_transformer.transform(X_new_counts)\n\npredicted = clf2.predict(X_new_tfidf)')

get_ipython().system('python mornApp/build_model.py')

get_ipython().run_cell_magic('writefile', 'mornApp/mornApp.py', 'from flask import Flask\nfrom flask import request\napp = Flask(__name__)\nfrom flask import render_template\n\n# Form page to submit text\n#============================================\n# create page with a form on it\n@app.route(\'/\')\ndef submission_page():\n    #content = \'hello\'\n    return render_template(\'template.html\')\n\n@app.route(\'/about\')\ndef about_page():\n    #content = \'hello\'\n    return render_template(\'about.html\')\n\n    \'\'\'\n    \'\'\'\n# <form action="/word_counter" method=\'POST\' >\n#         <input type="text" name="user_input" />\n#         <input type="submit" />\n#     </form>\n# My word counter app\n#==============================================\n# create the page the form goes to\n@app.route(\'/word_counter\', methods=[\'POST\',\'GET\'] )\ndef word_counter():\n#     if request.method == \'POST\':\n#         return \'\'\n    # get data from request form, the key is the name you set in your form\n    data = request.form[\'user_input\']\n\n    # convert data to list\n    data = [data]\n\n    import pickle\n    import pandas as pd\n    import matplotlib.pyplot as plt\n    import numpy as np\n    from sklearn.feature_extraction.text import CountVectorizer\n    from sklearn.feature_extraction.text import TfidfTransformer\n    from sklearn.naive_bayes import MultinomialNB\n    \n    clf2 = pickle.load(open( "mornApp/data/my_model.pkl", "rb" ) )\n    count_vect2 = pickle.load(open( "mornApp/data/my_vectorizer.pkl", "rb" ) )\n    tfidf_transformer2 = pickle.load(open ( "mornApp/data/my_transformer.pkl", "rb" ))\n\n    #process new data\n    X_new_counts = count_vect2.transform(data)\n    X_new_tfidf = tfidf_transformer2.transform(X_new_counts)\n    predicted = clf2.predict(X_new_tfidf)\n    \n    #output the category that the text is in\n    categories = [\'alt.atheism\', \'comp.graphics\', \'sci.med\', \'soc.religion.christian\']\n    for doc, category in zip(data, predicted):\n        #return(\'%r => %s\' % (doc, categories[category]))\n        return render_template(\'template2.html\', doc=doc, category=categories[category])\n\nif __name__ == \'__main__\':\n    app.run(host=\'0.0.0.0\', port=8087, debug=True)')

get_ipython().system('python mornApp/mornApp.py')

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np
pd.set_option('display.max_columns', 999)

categories = ['alt.atheism', 'soc.religion.christian',           'comp.graphics', 'sci.med']
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)

twenty_train.target_names 

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

import pickle
vectorizer_pickle = pickle.dumps(count_vect)
model_pickle = pickle.dumps(clf)
clf2 = pickle.loads(model_pickle)
count_vect2 = pickle.loads(vectorizer_pickle)

docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect2.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf2.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

