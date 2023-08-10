import pandas as pd
from tqdm import tqdm_notebook as tqdm
import random
from pandas import ExcelWriter

talk = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/talk_filtered_3.csv')
talk

thread_lens = talk.groupby(['article', 'thread']).size()
thread_lens

thresholds = [5, 10, 15, 20, 30, 40]
binned_threads = {min_len: list() for min_len in thresholds}

for min_len in thresholds:
    binned_threads[min_len] = thread_lens[thread_lens > min_len].index.tolist()

binned_threads

for min_len in thresholds:
    print(min_len, end=': ')
    print(len(binned_threads[min_len]))

# Sample from 15-threshold threads
sample = random.sample(binned_threads[15], 10)
sample

# Sample from 10-threshold threads
samples = {}
samples[10] = random.sample(binned_threads[10], 10)
samples[10]

# Sample from 5-threshold threads
t = 5
samples = {}
samples[t] = random.sample(binned_threads[t], 10)
samples[t]

pd.set_option('display.max_colwidth', 999)

n = 0
a,t = sample[n]
rows = talk[(talk['article']==a) & (talk['thread']==t)]
rows

# for a,t in sample:
#     rows = talk[(talk['article']==a) & (talk['thread']==t)]
#     print(rows)

selected = [
    ('Anton Balasingham', 'Asia Tribune vs. Tamilnet'),
    ('Polish occupation of Czechoslovakia', 'Extent of coverage of the postwar expulsions'),
    ('Priesthood keys', 'Priesthood (Mormonism) Chart'),
    ('Acharya S', 'Delete actual name'),
    ('Rubber tired metro', 'Rubber Tired Metros as burdensome and having only political (not technical) merit'),
    ('33550336 (number)', 'The first two lines in disagreement.'),
    ('Thermal light', 'Reworked'),
    ('Satan (South Park)', 'Proposed removal of content'),
    ('Mount Hebron', 'Removal of well sourced material'),
    ('Tour de France', 'Armstrong')
]
len(selected)

# Put in editor scores
scores = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/enwiki_talk_scores.csv')
scores.columns

merged = pd.merge(talk, scores, left_on=['article', 'thread', 'username'], right_on=['article', 'thread_title', 'editor'])
merged = merged[talk.columns.tolist() + ['editor_score']]
merged

cols = merged.columns.tolist()
new_cols = cols[:3] + ['username', 'timestamp'] + cols[5:]
merged = merged[new_cols]

mask = list(map(lambda tup: tup in selected, zip(merged['article'], merged['thread'])))
selected_rows = merged[mask]
selected_rows

selected_rows.to_csv('/home/michael/school/research/wp/jsalt_tutorial/examples.csv', index=False)

example_dfs = [merged[merged['article']==a] for a in [art for art,_ in selected]]
len(example_dfs)

writer = ExcelWriter('/home/michael/school/research/wp/jsalt_tutorial/wikipedia_talk_page_examples.xlsx')

for df in example_dfs:
    df.to_excel(writer, df['article'].unique().tolist()[0][:31], index=False)
    
writer.save()

