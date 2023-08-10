from os import path
import pandas
import seaborn as _
get_ipython().magic('matplotlib inline')

rome_version = 'v330'
data_folder = '../../../data'
rome_folder = path.join(data_folder, 'rome/csv')
mobility_csv = path.join(rome_folder, 'unix_rubrique_mobilite_%s_utf8.csv' % rome_version)
rome_csv = path.join(rome_folder, 'unix_referentiel_code_rome_%s_utf8.csv' % rome_version)
appellation_csv = path.join(rome_folder, 'unix_referentiel_appellation_%s_utf8.csv' % rome_version)

mobility = pandas.read_csv(mobility_csv)
rome = pandas.read_csv(rome_csv)[['code_rome', 'libelle_rome']]
rome_names = rome.groupby('code_rome').first()['libelle_rome']
jobs = pandas.read_csv(appellation_csv)[['code_ogr', 'code_rome', 'libelle_appellation_court']]
jobs_names = jobs.groupby('code_ogr').first()['libelle_appellation_court']

mobility.head(2).transpose()

mobility.count()

mobility[mobility.code_appellation_source.notnull()].head(2).transpose()

# Rename columns.
mobility.rename(columns={
        'code_rome': 'group_source',
        'code_appellation_source': 'job_source',
        'code_rome_cible': 'group_target',
        'code_appellation_cible': 'job_target',
    }, inplace=True)

# Add names.
mobility['group_source_name'] = mobility['group_source'].map(rome_names)
mobility['group_target_name'] = mobility['group_target'].map(rome_names)
mobility['job_source_name'] = mobility['job_source'].map(jobs_names)
mobility['job_target_name'] = mobility['job_target'].map(jobs_names)

# Sort columns.
mobility = mobility[[
        'group_source', 'group_source_name', 'job_source', 'job_source_name',
        'group_target', 'group_target_name', 'job_target', 'job_target_name',
        'code_type_mobilite', 'libelle_type_mobilite'
    ]]

mobility.head(2).transpose()

# Links from one job group to the same one.
len(mobility[mobility.group_source == mobility.group_target].index)

# Number of duplicate links.
len(mobility.index) - len(mobility.drop_duplicates())

# Number of duplicate links when we ignore the link types.
len(mobility.index) - len(mobility.drop_duplicates([
    'group_source', 'job_source', 'group_target', 'job_target']))

# Reverse links.
two_links = pandas.merge(
    mobility.fillna(''), mobility.fillna(''),
    left_on=['group_target', 'job_target'],
    right_on=['group_source', 'job_source'])
str(len(two_links[
        (two_links.group_source_x == two_links.group_target_y) &
        (two_links.job_source_x == two_links.job_target_y)].index) / len(mobility.index) * 100) + '%'

rome_froms = pandas.merge(
    mobility[mobility.job_source.notnull()].drop_duplicates(['group_source', 'group_source_name']),
    mobility[mobility.job_source.isnull()].drop_duplicates(['group_source', 'group_source_name']),
    on=['group_source', 'group_source_name'], how='outer', suffixes=['_specific', '_group'])

# Number of ROME job groups that have links both for the group and for at least one specific job.
len(rome_froms[rome_froms.group_target_specific.notnull() & rome_froms.group_target_group.notnull()])

# ROME job groups that have only links for specific jobs and not for the group.
rome_froms[rome_froms.group_target_group.isnull()]['group_source_name'].tolist()

rome_froms = pandas.merge(
    mobility[mobility.job_source.notnull()].drop_duplicates(['group_target', 'group_target_name']),
    mobility[mobility.job_source.isnull()].drop_duplicates(['group_target', 'group_target_name']),
    on=['group_target', 'group_target_name'], how='outer', suffixes=['_specific', '_group'])

# Number of ROME job groups that have links both to the group and to at least one specific job.
len(rome_froms[rome_froms.group_source_specific.notnull() & rome_froms.group_source_group.notnull()])

# ROME job groups that have only links to specific jobs and not to the whole group.
rome_froms[rome_froms.group_source_group.isnull()]['group_target_name'].tolist()

# Number of links specific to jobs (as opposed to groups) that are already specified by group links.
mobility['has_job_source'] = ~mobility.job_source.isnull()
mobility['has_job_target'] = ~mobility.job_target.isnull()
any_job_mobility = mobility.drop_duplicates(['group_source', 'has_job_source', 'group_target', 'has_job_target'])
len(any_job_mobility) - len(any_job_mobility.drop_duplicates(['group_source', 'group_target']))

# In this snippet, we count # of links to groups & to specific jobs for each job.

mobility_from_group = mobility[mobility.job_source.isnull()][['group_source', 'group_target', 'job_target']]
# Count # of groups that are linked from each group.
mobility_from_group['target_groups'] = (
    mobility_from_group[mobility_from_group.job_target.isnull()]
        .groupby('group_source')['group_source'].transform('count'))
mobility_from_group['target_groups'].fillna(0, inplace=True)
# Count # of specific jobs that are linked from each group.
mobility_from_group['target_jobs'] = (
    mobility_from_group[mobility_from_group.job_target.notnull()]
        .groupby('group_source')['group_source'].transform('count'))
mobility_from_group['target_jobs'].fillna(0, inplace=True)

mobility_from_group = mobility_from_group.groupby('group_source', as_index=False).max()[
    ['group_source', 'target_groups', 'target_jobs']]


mobility_from_job = mobility[mobility.job_source.notnull()][['job_source', 'group_target', 'job_target']]
# Count # of groups that are linked from each job.
mobility_from_job['target_groups'] = (
    mobility_from_job[mobility_from_job.job_target.isnull()]
        .groupby('job_source')['job_source'].transform('count'))
mobility_from_job['target_groups'].fillna(0, inplace=True)
# Count # of jobs that are linked from each job.
mobility_from_job['target_jobs'] = (
    mobility_from_job[mobility_from_job.job_target.notnull()]
        .groupby('job_source')['job_source'].transform('count'))
mobility_from_job['target_jobs'].fillna(0, inplace=True)

mobility_from_job = mobility_from_job.groupby('job_source', as_index=False).max()[
    ['job_source', 'target_groups', 'target_jobs']]

jobs_with_counts = pandas.merge(
    jobs, mobility_from_group, left_on='code_rome', right_on='group_source', how='left')
jobs_with_counts = pandas.merge(
    jobs_with_counts, mobility_from_job, left_on='code_ogr', right_on='job_source', how='left')

jobs_with_counts.fillna(0, inplace=True)
jobs_with_counts['target_groups'] = jobs_with_counts.target_groups_x + jobs_with_counts.target_groups_y
jobs_with_counts['target_jobs'] = jobs_with_counts.target_jobs_x + jobs_with_counts.target_jobs_y
jobs_with_counts['total'] = jobs_with_counts['target_groups'] + jobs_with_counts['target_jobs']

jobs_with_counts = jobs_with_counts[['code_ogr', 'libelle_appellation_court', 'target_groups', 'target_jobs', 'total']]

# Jobs that don't have any links from them or from their group.
jobs_with_counts[jobs_with_counts.total == 0]['libelle_appellation_court'].tolist()

jobs_with_counts.total.hist()
str(len(jobs_with_counts.total[jobs_with_counts.total >= 5].index) / len(jobs_with_counts.index)*100) + '%'

