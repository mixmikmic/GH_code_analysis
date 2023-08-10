# Click the Blue Plane to preview this notebook as a CrossCompute Tool
source_text_path = 'selected-reminders.txt'
word_count_per_concept = 7
concept_count = 3
target_folder = '/tmp'

import re
source_text = open(source_text_path, 'rt').read()
source_text = re.sub(r'[^a-zA-Z\s]', '', source_text)
source_text = source_text.lower()
words = source_text.split()

import random
concept_lines = []
for concept_index in range(concept_count):
    random_words = random.choices(words, k=word_count_per_concept)
    concept_lines.append(' '.join(random_words))
concept_lines

from os.path import join
target_path = join(target_folder, 'concepts.txt')
with open(target_path, 'wt') as target_file:
    target_file.write('\n'.join(concept_lines))
print('concepts_text_path = %s' % target_path)

