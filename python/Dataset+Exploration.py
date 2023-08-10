get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas
import os
import sys

import seaborn as sns
sns.set_style('white')
sns.set_palette('colorblind')

from collections import Counter

from imp import reload
import read_annotations
reload(read_annotations);

ANNOTATORS = {
    'C': {'dirname': 'C'}, 'L': {'dirname': 'L'}, 'M': {'dirname': 'M'}, 'S': {'dirname': 'S'}
}
ANNOTATION_DIR = os.path.join(os.path.expanduser('~'), 'am/data/echr/annotation/')
documents = read_annotations.get_all_documents(ANNOTATION_DIR, ANNOTATORS)

documents

stats_columns = ['Tokens', 'Labeled Tokens', 'Claims', 'Premises', 'Major claims',
                 'Components', 'Relations']
statistics = pandas.DataFrame(
    index=[document.identifier 
           for annotator_documents in documents.values() for document in annotator_documents],
    columns=['Annotator'] + stats_columns)

def get_statistics(document, annotator):
    id = document.identifier
    words, labels = document.get_word_label_list()
    statistics.loc[id]['Annotator'] = annotator
    statistics.loc[id]['Tokens'] = len(words)
    statistics.loc[id]['Labeled Tokens'] = len(
        [label for label in labels if label != document.default_label])
    counts = Counter(labels)
    statistics.loc[id]['Claims'] = counts['claim']
    statistics.loc[id]['Major claims'] = counts['major-claim']
    statistics.loc[id]['Premises'] = counts['premise']
    statistics.loc[id]['Components'] = len(document.annotated_components)
    statistics.loc[id]['Relations'] = len(document.annotated_relations)

for annotator, annotator_documents in documents.items():
    for document in annotator_documents:
        get_statistics(document, annotator)
statistics[stats_columns] = statistics[stats_columns].astype(int)

statistics

statistics.sum()

statistics.describe()

statistics.groupby('Annotator').describe()

doc = documents['S'][0]

type(doc)

sections = pandas.DataFrame(columns=['Document', 'Section', 'Sentences'])
for annotator, docs in documents.items():
    for doc in docs:
        for section_name, count in zip(*numpy.unique([sentence.section for sentence in doc.sentences],
                                                     return_counts=True)):
            sections.loc[sections.shape[0]] = {
                'Document': doc.identifier.replace('Case: ', ''), 'Section': section_name[:10], 'Sentences': count}
sections[:10]

sections.Section.unique()

sns.factorplot(data=sections, x='Sentences', y='Section', col='Document', kind="bar", col_wrap=3, orient="h")

relations = pandas.DataFrame(columns=['Document', 'Start', 'End', 'Label', 'Span'])
for annotator, docs in documents.items():
    for doc in docs:
        for start, info in doc.annotated_relations.items():
            for end, label in info.items():
                relations.loc[relations.shape[0]] = {
                    'Document': doc.identifier.replace('Case: ', ''),
                    'Start': start, 'End': end, 'Span': end-start,
                    'Label': label
                }
relations[['Document', 'Label', 'Start']].groupby(['Document', 'Label']).count()

sns.factorplot(data=relations, x='Label', kind="count")

sns.factorplot(data=relations, x='Label', kind="count", col='Document', col_wrap=4)

sns.factorplot(data=relations, x='Label', col='Document', y='Span', kind="violin", col_wrap=4)

labels = []
for annotator, docs in documents.items():
    for doc in docs:
        _, label = doc.get_word_label_list()
        labels.extend(label)
numpy.unique(labels, return_counts=True)

