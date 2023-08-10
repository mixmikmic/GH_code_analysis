import datetime as dt
import os
import sys

from cltk.corpus.greek.tlg.parse_tlg_indices import get_id_author
from cltk.corpus.utils.formatter import assemble_tlg_author_filepaths
from nltk.tokenize.punkt import PunktLanguageVars
import pandas

p = PunktLanguageVars()

t0 = dt.datetime.utcnow()

cleaned_dir = os.path.expanduser('~/cltk_data/greek/text/tlg/plaintext_clean')
dir_contents = os.listdir(cleaned_dir)

corpus_stats = {}

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

corpus_stats = {'doc_count': doc_count, 
               'total_words': len(all_tokens_list),
               'total_unique_words': len(all_tokens_unique)}

print('... finished in {}'.format(dt.datetime.utcnow() - t0))

df_corpus = pandas.DataFrame(corpus_stats, index=[0])
print(df_corpus)

from cltk.corpus.greek.tlg.parse_tlg_indices import get_epithet_of_author

map_id_author = get_id_author()

t0 = dt.datetime.utcnow()

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
#     print(author)
#     print('    ', 'Total words:', len(tokens))
#     print('    ', 'Total unique words:', len(set(tokens)))
#     print('    ', 'Lexical diversity:', lexical_diversity)

print('... finished in {}'.format(dt.datetime.utcnow() - t0))

df_text_counts = pandas.DataFrame(map_id_word_counts).T

df_text_counts

df_text_counts.to_csv(os.path.expanduser('~/cltk_data/user_data/stats_text_counts.csv'))

from statistics import mean
from statistics import stdev

author_stats = {}
corpus_word_count_all = []
corpus_word_count_unique = []
corpus_word_lexical_diversity = []
for author_id, map_counts in map_id_word_counts.items():
    corpus_word_count_all.append(map_counts['word_count_all'])
    corpus_word_count_unique.append(map_counts['word_count_unique'])
    corpus_word_lexical_diversity.append(map_counts['lexical_diversity'])

author_stats['mean_words_per_author'] = mean(corpus_word_count_all)
author_stats['standard_deviation_of_words_per_author:'] = stdev(corpus_word_count_all)
author_stats['mean_unique_words_per_author'] = mean(corpus_word_count_unique)
author_stats['standard_deviation_of_unique_words_per_author'] = stdev(corpus_word_count_unique)
author_stats['lexical_diversity_per_author'] = mean(corpus_word_lexical_diversity)
author_stats['standard_deviation_of_lexical_diversity_per_author:'] = stdev(corpus_word_lexical_diversity)

print('Mean words per author:', mean(corpus_word_count_all))
print('Standard deviation of words per author:', stdev(corpus_word_count_all))

print('Mean unique words per author:', mean(corpus_word_count_unique))
print('Standard deviation of unique words per author:', stdev(corpus_word_count_unique))

print('Lexical diversity per author:', mean(corpus_word_lexical_diversity))
print('Standard deviation of lexical diversity per author:', stdev(corpus_word_lexical_diversity))

df_authors = pandas.DataFrame(author_stats, index=[0])
df_authors

df_authors.to_csv(os.path.expanduser('~/cltk_data/user_data/stats_authors.csv'))

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
epithet_scores = {}
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
    print('    Standard deviation of lexical diversity:', ld_standard_deviation)
    
    epithet_lexical_diversity_tuples.append((epithet, mean(lexical_diversity_list)))

    tmp_scores = {}
    tmp_scores['mean_of_word_counts_ per_author'] = mean(counts)
    tmp_scores['standard_deviation_of_word_counts_per_author'] = wc_standard_deviation
    tmp_scores['mean_of_unique_word_counts_per_author'] = mean(uniques_list)
    tmp_scores['standard_deviation_of_unique_word_counts_per_author'] = uniques_standard_deviation
    tmp_scores['mean_of_lexical_diversity_per_author'] = mean(lexical_diversity_list)
    tmp_scores['standard_deviation_of_lexical_diversity'] = ld_standard_deviation
    epithet_scores[epithet] = tmp_scores

# sort epithets by lexical diversity
sorted(epithet_lexical_diversity_tuples, key=lambda x: x[1], reverse=True)

pandas.DataFrame(epithet_lexical_diversity_tuples)

df_epithet_scores = pandas.DataFrame(epithet_scores).T
df_epithet_scores

df_epithet_scores.to_csv(os.path.expanduser('~/cltk_data/user_data/stats_epithet.csv'))



