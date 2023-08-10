from rake_nltk import Rake
from collections import OrderedDict
from operator import itemgetter 

r = Rake()

with open('sample_text.txt', 'r') as f:
    text=f.read().replace('\n', '')

r.extract_keywords_from_text(text)

r.get_ranked_phrases()

r.get_ranked_phrases_with_scores()

word_degrees = r.get_word_degrees()

sorted_word_degrees = OrderedDict(sorted(word_degrees.items(), key = itemgetter(1), reverse = True))

sorted_word_degrees

word_frequency = r.frequency_dist

sorted_word_frequency = OrderedDict(sorted(word_frequency.items(), key = itemgetter(1), reverse = True))

sorted_word_frequency



