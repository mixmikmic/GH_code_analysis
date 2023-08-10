from sklearn.feature_extraction.text import TfidfVectorizer
from privacy_bot.analysis.policies_snapshot_api import Policies
from privacy_bot.analysis.visualization import wordcloud_from_dict

import json

policies = Policies()
corpus = [policy.text for policy in policies.query(lang='en')]

# Vectorizing policies
vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
X = vectorizer.fit_transform(corpus)
idf = vectorizer.idf_

# Getting scored words
idf_dict = dict(zip(vectorizer.get_feature_names(), idf))
sorted_word_idf_tuples = sorted(idf_dict.items(), key=lambda x: x[1])

# Look at top 20 most relevant words found
print([w for (w, idf) in sorted_word_idf_tuples[0:20]])

freq_idf_dict = dict((k, (1.0/v)*100) for k, v in idf_dict.items())

wordcloud_from_dict(freq_idf_dict)



