import graphlab

people = graphlab.SFrame('people_wiki.gl/')

people['word_count'] = graphlab.text_analytics.count_words(people['text'])

elton = people[people['name'] == 'Elton John']

elton['word_count'] = graphlab.text_analytics.count_words(elton['text'])

elton_word_count_table = elton[['word_count']].stack('word_count', new_column_name=['word', 'count'])

elton_word_count_table.sort('count', ascending=False)

tfidf = graphlab.text_analytics.tf_idf(people['word_count'])

people['tfidf'] = tfidf['docs']

elton2 = people[people['name'] == 'Elton John']

elton_tfidf_talbe = elton2[['tfidf']].stack('tfidf', new_column_name=['word', 'tfidf'])

elton_tfidf_talbe.sort('tfidf', ascending=False)

paul = people[people['name'] == 'Paul McCartney']

victoria = people[people['name'] == 'Victoria Beckham']

graphlab.distances.cosine(elton2['tfidf'][0], paul['tfidf'][0])

graphlab.distances.cosine(elton2['tfidf'][0], victoria['tfidf'][0])

word_count_knn_model = graphlab.nearest_neighbors.create(people, features=['word_count'], distance='cosine', label='name')

tfidf_knn_model = graphlab.nearest_neighbors.create(people, features=['tfidf'], distance='cosine', label='name')

word_count_knn_model.query(elton2)

tfidf_knn_model.query(elton2)

word_count_knn_model.query(victoria)

tfidf_knn_model.query(victoria)

