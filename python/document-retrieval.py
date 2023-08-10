import graphlab

people = graphlab.SFrame('people_wiki.gl/')

people.head()

len(people)

obama = people[people['name'] == 'Barack Obama']

obama

obama['text']

clooney = people[people['name'] == 'George Clooney']
clooney['text']

obama['word_count'] = graphlab.text_analytics.count_words(obama['text'])

print obama['word_count']

obama_word_count_table = obama[['word_count']].stack('word_count', new_column_name = ['word','count'])

obama_word_count_table.head()

obama_word_count_table.sort('count',ascending=False)

people['word_count'] = graphlab.text_analytics.count_words(people['text'])
people.head()

tfidf = graphlab.text_analytics.tf_idf(people['word_count'])

# Earlier versions of GraphLab Create returned an SFrame rather than a single SArray
# This notebook was created using Graphlab Create version 1.7.1
if graphlab.version <= '1.6.1':
    tfidf = tfidf['docs']

tfidf

people['tfidf'] = tfidf

obama = people[people['name'] == 'Barack Obama']

obama[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)

clinton = people[people['name'] == 'Bill Clinton']

beckham = people[people['name'] == 'David Beckham']

obama['tfidf'][0]

graphlab.distances.cosine(obama['tfidf'][0],clinton['tfidf'][0])

graphlab.distances.cosine(obama['tfidf'][0],beckham['tfidf'][0])

knn_model = graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name')

knn_model.query(obama)

swift = people[people['name'] == 'Taylor Swift']

knn_model.query(swift)

jolie = people[people['name'] == 'Angelina Jolie']

knn_model.query(jolie)

arnold = people[people['name'] == 'Arnold Schwarzenegger']

knn_model.query(arnold)



