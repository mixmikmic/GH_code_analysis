get_ipython().run_cell_magic('capture', '', '%load_ext autoreload\n%autoreload 2\n%cd ..\nimport statnlpbook.tokenization as tok')

text = "Mr. Bob Dobolina is thinkin' of a master plan." +        "\nWhy doesn't he quit?"
text.split(" ")

import re
gap = re.compile('\s')
gap.split(text)

token = re.compile('\w+|[.?:]')
token.findall(text)

token = re.compile('Mr.|[\w\']+|[.?]')
tokens = token.findall(text)
tokens

jap = "彼は音楽を聞くのが大好きです"
re.compile('彼|は|く|音楽|を|聞くの|が|大好き|です').findall(jap)

tok.sentence_segment(re.compile('\.'), tokens)

