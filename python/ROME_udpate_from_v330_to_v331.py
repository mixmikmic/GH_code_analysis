import collections
import glob
import os
from os import path

import matplotlib_venn
import pandas

rome_path = path.join(os.getenv('DATA_FOLDER'), 'rome/csv')

OLD_VERSION = '330'
NEW_VERSION = '331'

old_version_files = frozenset(glob.glob(rome_path + '/*%s*' % OLD_VERSION))
new_version_files = frozenset(glob.glob(rome_path + '/*%s*' % NEW_VERSION))

new_files = new_version_files - frozenset(f.replace(OLD_VERSION, NEW_VERSION) for f in old_version_files)
deleted_files = old_version_files - frozenset(f.replace(NEW_VERSION, OLD_VERSION) for f in new_version_files)

print('%d new files' % len(new_files))
print('%d deleted files' % len(deleted_files))

new_to_old = dict((f, f.replace(NEW_VERSION, OLD_VERSION)) for f in new_version_files)

# Load all datasets.
Dataset = collections.namedtuple('Dataset', ['basename', 'old', 'new'])
data = [Dataset(
        basename=path.basename(f),
        old=pandas.read_csv(f.replace(NEW_VERSION, OLD_VERSION)),
        new=pandas.read_csv(f))
    for f in sorted(new_version_files)]

def find_dataset_by_name(data, partial_name):
    for dataset in data:
        if 'unix_%s_v%s_utf8.csv' % (partial_name, NEW_VERSION) == dataset.basename:
            return dataset
    raise ValueError('No dataset named %s, the list is\n%s' % (partial_name, [dataset.basename for d in data]))

for dataset in data:
    if set(dataset.old.columns) != set(dataset.new.columns):
        print('Columns of %s have changed.' % dataset.basename)

untouched = 0
for dataset in data:
    diff = len(dataset.new.index) - len(dataset.old.index)
    if diff > 0:
        print('%d values added in %s' % (diff, dataset.basename))
    elif diff < 0:
        print('%d values removed in %s' % (diff, dataset.basename))
    else:
        untouched += 1
print('%d/%d files with the same number of rows' % (untouched, len(data)))

items = find_dataset_by_name(data, 'item')

new_items = set(items.new.code_ogr) - set(items.old.code_ogr)
obsolete_items = set(items.old.code_ogr) - set(items.new.code_ogr)
stable_items = set(items.new.code_ogr) & set(items.old.code_ogr)

_ = matplotlib_venn.venn2((len(obsolete_items), len(new_items), len(stable_items)), (OLD_VERSION, NEW_VERSION))

items.old[items.old.code_ogr.isin(obsolete_items)].tail()

items.new[items.new.code_ogr.isin(new_items)].head()

links = find_dataset_by_name(data, 'liens_rome_referentiels')
old_links_on_stable_items = links.old[links.old.code_ogr.isin(stable_items)]
new_links_on_stable_items = links.new[links.new.code_ogr.isin(stable_items)]

old = old_links_on_stable_items[['code_rome', 'code_ogr']]
new = new_links_on_stable_items[['code_rome', 'code_ogr']]

links_merged = old.merge(new, how='outer', indicator=True)
links_merged['_diff'] = links_merged._merge.map({'left_only': 'removed', 'right_only': 'added'})
links_merged._diff.value_counts()

job_group_names = find_dataset_by_name(data, 'referentiel_code_rome').old.set_index('code_rome').libelle_rome
item_names = items.new.set_index('code_ogr').libelle.drop_duplicates()
links_merged['job_group_name'] = links_merged.code_rome.map(job_group_names)
links_merged['item_name'] = links_merged.code_ogr.map(item_names)
links_merged.dropna().head(10)

