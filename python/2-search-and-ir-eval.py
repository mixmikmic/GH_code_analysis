import metapy
idx = metapy.index.make_inverted_index('apnews-config.toml')

idx.num_docs()

idx.unique_terms()

idx.avg_doc_length()

idx.total_corpus_terms()

ranker = metapy.index.OkapiBM25()

query = metapy.index.Document()
query.content('Airbus Subsidies') # query from AP news

top_docs = ranker.score(idx, query, num_results=5)
top_docs

for num, (d_id, _) in enumerate(top_docs):
    content = idx.metadata(d_id).get('content')
    print("{}. {}...\n".format(num + 1, content[0:250]))

ev = metapy.index.IREval('apnews-config.toml')

num_results = 10
with open('queries.txt') as query_file:
    for query_num, line in enumerate(query_file):
        query.content(line.strip())
        results = ranker.score(idx, query, num_results)                            
        avg_p = ev.avg_p(results, query_num, num_results)
        print("Query {} average precision: {}".format(query_num + 1, avg_p))

ev.map()

class SimpleRanker(metapy.index.RankingFunction):                                            
    """                                                                          
    Create a new ranking function in Python that can be used in MeTA.             
    """                                                                          
    def __init__(self, some_param=1.0):                                             
        self.param = some_param
        # You *must* call the base class constructor here!
        super(SimpleRanker, self).__init__()                                        
                                                                                 
    def score_one(self, sd):
        """
        You need to override this function to return a score for a single term.
        For fields available in the score_data sd object,
        @see https://meta-toolkit.org/doxygen/structmeta_1_1index_1_1score__data.html
        """
        return (self.param + sd.doc_term_count) / (self.param * sd.doc_unique_terms + sd.doc_size)

