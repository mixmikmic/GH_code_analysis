sentence = "ΑΥΤΟΣ είναι ο χορός της βροχής της φυλής, ό,τι περίεργο."
sentence = sentence.lower()
sentence

from unidecode import unidecode

sentence_latin = unidecode(sentence)
sentence_latin

import unicodedata

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) # NFD = Normalization Form Canonical Decomposition, one of four Unicode normalization forms.
                   if unicodedata.category(c) != 'Mn') # The character category "Mn" stands for Nonspacing_Mark
sentence_no_accents = strip_accents(sentence)
sentence_no_accents

from nltk.tokenize import WhitespaceTokenizer

tokens = WhitespaceTokenizer().tokenize(sentence_no_accents)
tokens

from string import punctuation

new_tokens = []

for token in tokens:
    if token == 'ο,τι':
        new_tokens.append('ο,τι')
    else:
        new_tokens.append(token.translate(str.maketrans({key: None for key in punctuation})))

new_tokens_with_stopwords = new_tokens
new_tokens

# Greek stopwords adapted from https://github.com/6/stopwords-json however better lists with more stopwords are available: https://www.translatum.gr/forum/index.php?topic=3550.0?topic=3550.0
greek_stopwords = ["αλλα","αν","αντι","απο","αυτα","αυτες","αυτη","αυτο","αυτοι","αυτος","αυτους","αυτων","για","δε","δεν","εαν","ειμαι","ειμαστε","ειναι","εισαι","ειστε","εκεινα","εκεινες","εκεινη","εκεινο","εκεινοι","εκεινος","εκεινους","εκεινων","ενω","επι","η","θα","ισως","κ","και","κατα","κι","μα","με","μετα","μη","μην","να","ο","οι","ομως","οπως","οσο","οτι","ο,τι","παρα","ποια","ποιες","ποιο","ποιοι","ποιος","ποιους","ποιων","που","προς","πως","σε","στη","στην","στο","στον","στης","στου","στους","στις","στα","τα","την","της","το","τον","τοτε","του","των","τις","τους","ως"]
len(greek_stopwords)

new_tokens_set = set(new_tokens)
greek_stopwords_set = set(greek_stopwords)
intersection_set = new_tokens_set.intersection(greek_stopwords_set)
intersection_set

for element in intersection_set:
    new_tokens = list(filter((element).__ne__, new_tokens)) # __ne__ is the != operator.
new_tokens

