from stanfordcorenlp import StanfordCoreNLP

# 'localhost' does not work inside container - use local ip address
corenlp_server = 'http://192.168.178.20:9000/'

url = corenlp_server.replace('/','').split(':')
host, port = ('://'.join(url[0:2]), int(url[2]))

nlp = StanfordCoreNLP(host, port=port)

props = {'annotators': 'tokenize,ssplit,pos'}
print(nlp.annotate(u'Köln is a city in Germany.', properties=props))

