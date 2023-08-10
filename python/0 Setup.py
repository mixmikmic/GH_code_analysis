import datetime as dt
from cltk.corpus.utils.importer import CorpusImporter

corpus_importer = CorpusImporter('greek')

corpus_importer.list_corpora

corpus_importer.import_corpus('tlg', '/root/classics_corpora/TLG_E')

from cltk.corpus.greek.tlgu import TLGU

corpus_importer.import_corpus('greek_software_tlgu')

t = TLGU()

t0 = dt.datetime.utcnow()

t.convert_corpus(corpus='tlg')

print('... finished in {}'.format(dt.datetime.utcnow() - t0))

with open('/root/cltk_data/greek/text/tlg/plaintext/TLG0007.TXT') as file_open:
    text_snippet = file_open.read()[:1500]
print(text_snippet)



