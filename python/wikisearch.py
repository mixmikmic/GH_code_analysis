import nbimporter
from corpora import FileStream
from benchmark import Wikisearch
import wikipedia

outfolder = '/Users/alfio/Dati/wikisearch/brat_20'
infolder = '/Users/alfio/Research/NCSR/argumentmining/corpora/brat/brat-project/brat-project'

W = Wikisearch(outfolder)
corpus = FileStream(infolder, file_ext='txt')

for i, doc in enumerate(corpus.docs[:20]):
    q = corpus.first_line(doc)
    q = q.rstrip('\n').lower()
    print 'searching for', q
    try:
        W.search(q)
        print sum([len(y) for x, y in W.mapping.items()]), 'documents retrieved'
    except wikipedia.exceptions.PageError:
        print sum([len(y) for x, y in W.mapping.items()]), 'page error'
    except wikipedia.exceptions.DisambiguationError:
        print sum([len(y) for x, y in W.mapping.items()]), 'page error'

W.save(content=False)



