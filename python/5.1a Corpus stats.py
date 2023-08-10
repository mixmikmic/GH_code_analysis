import os
import sys

from cltk.corpus.greek.tlg.parse_tlg_indices import get_id_author
from cltk.corpus.utils.formatter import assemble_tlg_author_filepaths
from nltk.tokenize.punkt import PunktLanguageVars

p = PunktLanguageVars()

cleaned_dir = os.path.expanduser('~/cltk_data/greek/text/tlg/plaintext_clean')
dir_contents = os.listdir(cleaned_dir)

all_tokens_list = []

for doc_count, file in enumerate(dir_contents):
    file_path = os.path.join(cleaned_dir, file)
    with open(file_path) as fo:
        text = fo.read().lower()
    text = ''.join([char for char in text if char not in ['.']])
    tokens = p.word_tokenize(text)
    all_tokens_list += tokens

print('Total author files:', doc_count)
print('Total words:', len(all_tokens_list))
all_tokens_unique = set(all_tokens_list)
print('Total unique words:', len(all_tokens_unique))

from cltk.corpus.greek.tlg.parse_tlg_indices import get_epithet_of_author

map_id_author = get_id_author()

# Words and unique words per author
map_id_word_counts = {}
for file in dir_contents:
    map_word_counts = {}
    file_path = os.path.join(cleaned_dir, file)
    author_id = file[3:-4]
    author = map_id_author[author_id]
    with open(file_path) as fo:
        text = fo.read().lower()
    text = ''.join([char for char in text if char not in ['.']])
    tokens = p.word_tokenize(text)
    map_word_counts['name'] = author
    map_word_counts['epithet'] = get_epithet_of_author(author_id)
    map_word_counts['word_count_all'] = len(tokens)
    map_word_counts['word_count_unique'] = len(set(tokens))
    try:
        lexical_diversity = len(set(tokens)) / len(tokens)
    except ZeroDivisionError:
        lexical_diversity = 0
    map_word_counts['lexical_diversity'] = lexical_diversity
    
    map_id_word_counts[author_id] = map_word_counts
    print(author)
    print('    ', 'Total words:', len(tokens))
    print('    ', 'Total unique words:', len(set(tokens)))
    print('    ', 'Lexical diversity:', lexical_diversity)

from statistics import mean
from statistics import stdev

corpus_word_count_all = []
corpus_word_count_unique = []
corpus_word_lexical_diversity = []
for author_id, map_counts in map_id_word_counts.items():
    corpus_word_count_all.append(map_counts['word_count_all'])
    corpus_word_count_unique.append(map_counts['word_count_unique'])
    corpus_word_lexical_diversity.append(map_counts['lexical_diversity'])

print('Mean words per author:', mean(corpus_word_count_all))
print('Standard deviation of words per author:', stdev(corpus_word_count_all))

print('Mean unique words per author:', mean(corpus_word_count_unique))
print('Standard deviation of unique words per author:', stdev(corpus_word_count_unique))

print('Lexical diversity per author:', mean(corpus_word_lexical_diversity))
print('Standard deviation of lexical diversity per author:', stdev(corpus_word_lexical_diversity))

from collections import defaultdict
import datetime as dt
from cltk.corpus.greek.tlg.parse_tlg_indices import get_epithets

list_epithets = get_epithets()

t0 = dt.datetime.utcnow()

map_epithet_counts_all = defaultdict(list)
map_epithet_counts_unique = defaultdict(list)
map_epithet_lexical_diversity = defaultdict(list)
for file in dir_contents:
    map_word_counts = defaultdict(list)
    file_path = os.path.join(cleaned_dir, file)
    author_id = file[3:-4]
    author = map_id_author[author_id]
    with open(file_path) as fo:
        text = fo.read().lower()
    text = ''.join([char for char in text if char not in ['.']])
    tokens = p.word_tokenize(text)
    try:
        lexical_diversity = len(set(tokens)) / len(tokens)
    except ZeroDivisionError:
        lexical_diversity = 0
    epithet = get_epithet_of_author(author_id)

    map_epithet_counts_all[epithet].append(len(tokens))
    map_epithet_counts_unique[epithet].append(len(set(tokens)))
    map_epithet_lexical_diversity[epithet].append(lexical_diversity)

print('... finished in {}'.format(dt.datetime.utcnow() - t0))

from statistics import StatisticsError

epithet_lexical_diversity_tuples = []
for epithet, counts in map_epithet_counts_all.items():
    print(epithet)
    print('    Mean of word counts per author:', mean(counts))
    try:
        wc_standard_deviation = stdev(counts)
    except StatisticsError:
        wc_standard_deviation = 0
    print('    Standard deviation of word counts per author:', wc_standard_deviation)
    
    uniques_list = map_epithet_counts_unique[epithet]
    print('    Mean of unique word counts per author:', mean(uniques_list))
    try:
        uniques_standard_deviation = stdev(uniques_list)
    except StatisticsError:
        uniques_standard_deviation = 0
    print('    Standard deviation of unique word counts per author:', uniques_standard_deviation)

    lexical_diversity_list = map_epithet_lexical_diversity[epithet]
    print('    Mean of lexical diversity per author:', mean(lexical_diversity_list))
    try:
        ld_standard_deviation = stdev(lexical_diversity_list)
    except StatisticsError:
        ld_standard_deviation = 0
    print('    Standard deviation of unique word counts per author:', ld_standard_deviation)
    
    epithet_lexical_diversity_tuples.append((epithet, mean(lexical_diversity_list)))

# sort epithets by lexical diversity
sorted(epithet_lexical_diversity_tuples, key=lambda x: x[1], reverse=True)

import pandas

df = pandas.DataFrame(epithet_lexical_diversity_tuples)
df



