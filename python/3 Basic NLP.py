# Let's start by binding a text string to a variable

# intro to Cato's de agricultura
cato_agri_praef = "Est interdum praestare mercaturis rem quaerere, nisi tam periculosum sit, et item foenerari, si tam honestum. Maiores nostri sic habuerunt et ita in legibus posiverunt: furem dupli condemnari, foeneratorem quadrupli. Quanto peiorem civem existimarint foeneratorem quam furem, hinc licet existimare. Et virum bonum quom laudabant, ita laudabant: bonum agricolam bonumque colonum; amplissime laudari existimabatur qui ita laudabatur. Mercatorem autem strenuum studiosumque rei quaerendae existimo, verum, ut supra dixi, periculosum et calamitosum. At ex agricolis et viri fortissimi et milites strenuissimi gignuntur, maximeque pius quaestus stabilissimusque consequitur minimeque invidiosus, minimeque male cogitantes sunt qui in eo studio occupati sunt. Nunc, ut ad rem redeam, quod promisi institutum principium hoc erit."

print(cato_agri_praef)

# http://docs.cltk.org/en/latest/latin.html#sentence-tokenization

from cltk.tokenize.sentence import TokenizeSentence

tokenizer = TokenizeSentence('latin')

cato_sentence_tokens = tokenizer.tokenize_sentences(cato_agri_praef)

print(cato_sentence_tokens)

# This has correctly identified 9 sentences
print(len(cato_sentence_tokens))

# viewed another way
for sentence in cato_sentence_tokens:
    print(sentence)
    print()

# import general-use word tokenizer
from cltk.tokenize.word import nltk_tokenize_words

cato_word_tokens = nltk_tokenize_words(cato_agri_praef)

print(cato_word_tokens)

cato_word_tokens_no_punt = [token for token in cato_word_tokens if token not in ['.', ',', ':', ';']]

print(cato_word_tokens_no_punt)

# number words
print(len(cato_word_tokens_no_punt))

# the set() function removes duplicates from a list
# let's see how many unique words are in here
cato_word_tokens_no_punt_unique = set(cato_word_tokens_no_punt)
print(cato_word_tokens_no_punt_unique)

print(len(cato_word_tokens_no_punt_unique))

# there's a mistake here though
# capitalized words ('At', 'Est', 'Nunc') would be counted incorrectly
# so let's lower the input string and try again
cato_agri_praef_lowered = cato_agri_praef.lower()
cato_word_tokens_lowered = nltk_tokenize_words(cato_agri_praef_lowered)

# now see all lowercase
print(cato_word_tokens_lowered)

# now let's do everything again
cato_word_tokens_no_punt_lowered = [token for token in cato_word_tokens_lowered if token not in ['.', ',', ':', ';']]
cato_word_tokens_no_punt_unique_lowered = set(cato_word_tokens_no_punt_lowered)
print(len(cato_word_tokens_no_punt_unique_lowered))

from cltk.tokenize.word import WordTokenizer
word_tokenizer = WordTokenizer('latin')
cato_cltk_word_tokens = word_tokenizer.tokenize(cato_agri_praef_lowered)
cato_cltk_word_tokens_no_punt = [token for token in cato_cltk_word_tokens if token not in ['.', ',', ':', ';']]

# now you can see the word '-que'
print(cato_cltk_word_tokens_no_punt)

# more total words
print(len(cato_cltk_word_tokens_no_punt))  # was 109

# more accurate unique words
cato_cltk_word_tokens_no_punt_unique = set(cato_cltk_word_tokens_no_punt)
print(len(cato_cltk_word_tokens_no_punt_unique))  # balances out to be the same (90)

# .difference() is an easy way to compare two sets
cato_cltk_word_tokens_no_punt_unique.difference(cato_word_tokens_no_punt_unique_lowered)

from cltk.stem.latin.j_v import JVReplacer
j = JVReplacer()
replaced_text = j.replace('vem jam')
print(replaced_text)

# let's start with the easiest method, which is to use Python's internal Counter()
from collections import Counter

# don't give the unique variation, but count all tokens
cato_word_counts_counter = Counter(cato_cltk_word_tokens_no_punt)
print(cato_word_counts_counter)

# the data structure of cato_word_counts_counter is a 'dictionary' in Python
# get the frequency of particular words like this:
print(cato_word_counts_counter['et'])

print(cato_word_counts_counter['qui'])

print(cato_word_counts_counter['maiores'])

# lex diversity of this little paragraph
print(len(cato_cltk_word_tokens_no_punt_unique) / len(cato_cltk_word_tokens_no_punt))
# meaning this is the ratio of unique to re-reused words

# of suriving word counts
from IPython.display import Image
Image('images/tableau_bubble.png')

Image('images/lexical_diversity_greek_canon.png')

# http://docs.cltk.org/en/latest/latin.html#stopword-filtering

# easist way to do this in Python is to use a list comprehension to remove stopwords

from cltk.stop.latin.stops import STOPS_LIST

print(STOPS_LIST)

cato_no_stops = [w for w in cato_cltk_word_tokens_no_punt if not w in STOPS_LIST]
# observe no stopwords
#! consider others you might want to add to the Latin stops list
print(cato_no_stops)

