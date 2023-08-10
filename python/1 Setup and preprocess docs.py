from cltk.corpus.utils.importer import CorpusImporter

my_greek_downloader = CorpusImporter('greek')

my_greek_downloader.import_corpus('tlg', '~/cltk/corpora/TLG_E/')

from cltk.corpus.greek.tlgu import TLGU

tlgu = TLGU()
tlgu.convert_corpus(corpus='tlg')  # writes to: ~/cltk_data/greek/text/tlg/plaintext/

get_ipython().system('head ~/cltk_data/greek/text/tlg/plaintext/TLG0437.TXT')

from cltk.corpus.utils.formatter import tlg_plaintext_cleanup
import os

plaintext_dir = os.path.expanduser('~/cltk_data/greek/text/tlg/plaintext/')
files = os.listdir(plaintext_dir)

for file in files:
    file = os.path.join(plaintext_dir, file)
    with open(file) as file_open:
        file_read = file_open.read()

    clean_text = tlg_plaintext_cleanup(file_read, rm_punctuation=True, rm_periods=False)
    clean_text = clean_text.lower()
    with open(file, 'w') as file_open:
        file_open.write(clean_text)

get_ipython().system('head ~/cltk_data/greek/text/tlg/plaintext/TLG0437.TXT ')

