from pycorenlp import StanfordCoreNLP

# 'localhost' does not work inside container - use local ip address
corenlp_server = 'http://192.168.178.20:9000/'

nlp = StanfordCoreNLP(corenlp_server)

props = {'annotators': 'tokenize,ssplit,pos'}
print(nlp.annotate(u'Köln is a city in Germany.', properties=props))

