import collections
import glob
import os
from os import path

import matplotlib_venn
import pandas

rome_path = path.join(os.getenv('DATA_FOLDER'), 'rome/csv')

OLD_VERSION = '329'
NEW_VERSION = '330'

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
        if partial_name in dataset.basename:
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

jobs = find_dataset_by_name(data, 'referentiel_appellation')
new_ogrs = set(jobs.new.code_ogr) - set(jobs.old.code_ogr)
new_jobs = jobs.new[jobs.new.code_ogr.isin(new_ogrs)]

job_groups = find_dataset_by_name(data, 'referentiel_code_rome_v')
pandas.merge(new_jobs, job_groups.new[['code_rome', 'libelle_rome']], on='code_rome', how='left')

competences = find_dataset_by_name(data, 'referentiel_competence')
new_ogrs = set(competences.new.code_ogr) - set(competences.old.code_ogr)
obsolete_ogrs = set(competences.old.code_ogr) - set(competences.new.code_ogr)
stable_ogrs = set(competences.new.code_ogr) & set(competences.old.code_ogr)

_ = matplotlib_venn.venn2((len(obsolete_ogrs), len(new_ogrs), len(stable_ogrs)), (OLD_VERSION, NEW_VERSION))

new_names = set(competences.new.libelle_competence) - set(competences.old.libelle_competence)
obsolete_names = set(competences.old.libelle_competence) - set(competences.new.libelle_competence)
stable_names = set(competences.new.libelle_competence) & set(competences.old.libelle_competence)

_ = matplotlib_venn.venn2((len(obsolete_names), len(new_names), len(stable_names)), (OLD_VERSION, NEW_VERSION))

print('Some skills that got removed: %s…' % ', '.join(sorted(obsolete_names)[:5]))
print('Some skills that got added: %s…' % ', '.join(sorted(new_names)[:5]))
print('Some skills that stayed: %s…' % ', '.join(sorted(stable_names)[:5]))

