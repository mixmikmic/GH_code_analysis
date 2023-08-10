raw = """'When I'M a Duchess,' she said to herself, (not in a very hopeful tone
though), 'I won't have any pepper in my kitchen AT ALL. Soup does very
well without--Maybe it's always pepper that makes people hot-tempered,'..."""

from nltk import re
sp = re.split(r' ', raw)
print(sp)

osp = re.split(r'[ \t\n]+', raw)
print(osp)

res = re.split(r'\s+', raw)
# built-in re abbreviation
print(res)

pycs = re.split(r'\W+', raw)
# Python character class
print(pycs)

'xx'.split('x')

pycs2 = re.findall(r'\w+', raw)
# w = word characters
print(pycs2)

exs = re.findall(r'\w+|\S\w*', raw)
# \S = non-whitespace characters
print(exs)

exs2 = re.findall(r"\w+(?:[-']\w+)*", raw)
print(exs2)

''' this '''
exs2 = re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", raw)
# [-.(]+
print(exs2)

exs2 = re.findall(r"\w+(?:[-']\w+)*|[-.(]+", raw)
# [-.(]+
print(exs2)

exs2 = re.findall(r"\w+(?:[-']\w+)*|[-(]+", raw)
# [-.(]+
print(exs2)

import nltk
text = 'That U.S.A. poster-print costs $12.40...'
pattern = r"""(?x)    # set flag to allow verbose regexps
              (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A.
              |\$?\d+(?:\.\d+)?%?       # numbers, incl. currency and percentages
              |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe
              |(?:[+/\-@&*])         # special characters with meanings
"""
nltk.regexp_tokenize(text, pattern)

from nltk.tokenize.regexp import RegexpTokenizer
tokeniser=RegexpTokenizer(pattern)
tokeniser.tokenize(text)



