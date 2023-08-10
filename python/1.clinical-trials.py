import bz2
import collections
import itertools
import os
import random
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree
import zipfile

import pandas

# # Uncomment this cell to download all trials as XML
# query = {'resultsxml': 'true'}
# query_str = urllib.parse.urlencode(query)
# query_url = 'http://clinicaltrials.gov/search?{}'.format(query_str) 
# zip_path = 'download/resultsxml.zip'
# urllib.request.urlretrieve(query_url, zip_path)

def zip_reader(path, max_records=None):
    """
    Generate study records from the bulk ClinicalTrials.gov XML zipfile.
    """
    with zipfile.ZipFile(path) as open_zip:
        filenames = open_zip.namelist()
        for i, filename in enumerate(filenames):
            with open_zip.open(filename) as open_xml:
                yield filename, xml.etree.ElementTree.parse(open_xml)
            if max_records is not None and i + 1 >= max_records:
                break

# %%time
# # Uncomment to prepare sample xml files
# random.seed(0)
# path = 'download/resultsxml.zip'
# for filename, tree in zip_reader(path):
#     if random.random() < 1e-4:
#         sample_path = os.path.join('download', 'sample', filename)
#         tree.write(sample_path)

# dhimmel/mesh commit
commit = '9e16dfdca6c6d32cf8d1dcb4149c86be58a1a029'

# Read MeSH descriptor and supplementary terms
url = 'https://github.com/dhimmel/mesh/blob/{}/data/descriptor-terms.tsv?raw=true'.format(commit)
desc_df = pandas.read_table(url)

url = 'https://github.com/dhimmel/mesh/blob/{}/data/supplemental-terms.tsv?raw=true'.format(commit)
supp_df = pandas.read_table(url)

assert not set(desc_df.TermName) & set(supp_df.TermName)

# Create a dictionary of MeSH term names to unique identifiers
mesh_name_to_id = dict(zip(desc_df.TermName, desc_df.DescriptorUI))
mesh_name_to_id.update(dict(zip(supp_df.TermName, supp_df.SupplementalRecordUI)))

unmatched_terms = collections.Counter()

def get_mesh_id(name):
    # Match by name
    mesh_id = mesh_name_to_id.get(name)
    if mesh_id is not None:
        return mesh_id
    # Match by name with first letter lowercase
    first_lower = name[0].lower() + name[1:]
    mesh_id = mesh_name_to_id.get(first_lower)
    if mesh_id is not None:
        return mesh_id
    # Return `None` for unmatched
    unmatched_terms[name] += 1
    return None

def get_mesh_ids(names):
    mesh_ids = [get_mesh_id(name) for name in names]
    return [x for x in mesh_ids if x is not None]

def parse_study_xml(tree):
    """
    Extract information from an element tree for a ClinicalTrials.gov XML record.
    """
    study = collections.OrderedDict()
    study['nct_id'] = tree.findtext('id_info/nct_id')
    study['study_type'] = tree.findtext('study_type')
    study['brief_title'] = tree.findtext('brief_title')
    brief_summary = tree.findtext('brief_summary/textblock', '')
    study['brief_summary'] = re.sub(r' *\n *', ' ', brief_summary).strip()
    study['overall_status'] = tree.findtext('overall_status')
    study['start_date'] = tree.findtext('start_date')
    study['phase'] = tree.findtext('phase')
    study['conditions'] = [x.text for x in tree.findall('condition')]
    study['intervention_drugs'] = [x.text for x in tree.findall('intervention[intervention_type="Drug"]/intervention_name')]
    study['mesh_conditions'] = get_mesh_ids(x.text for x in tree.findall('condition_browse/mesh_term'))
    study['mesh_interventions'] = get_mesh_ids(x.text for x in tree.findall('intervention_browse/mesh_term'))
    return study

get_ipython().run_cell_magic('time', '', "studies = list()\npath = 'download/resultsxml.zip'\nfor filename, tree in zip_reader(path):\n    study = parse_study_xml(tree)\n    studies.append(study)")

unmatched_terms.most_common(5)

study_df = pandas.DataFrame(studies)
study_df = study_df[list(studies[0].keys())]
study_df.tail(3)

# Save clinical trials, pipe delimiting plural fields
write_df = study_df.copy()
plural_columns = ['conditions', 'intervention_drugs', 'mesh_conditions', 'mesh_interventions']
for column in plural_columns:
    write_df[column] = write_df[column].map(lambda x: '|'.join(x))

with bz2.open('data/results.tsv.bz2', 'wt') as write_file:
    write_df.to_csv(write_file, sep='\t', index=False)

mesh_rows = list()
for study in studies:
    nct_id = study['nct_id']
    intervention = study['mesh_interventions']
    condition = study['mesh_conditions']
    for intervention, condition in itertools.product(intervention, condition):
        row = nct_id, intervention, condition
        mesh_rows.append(row)
mesh_df = pandas.DataFrame(mesh_rows, columns=['nct_id', 'intervention', 'condition'])
mesh_df = mesh_df.sort_values(['nct_id', 'intervention', 'condition'])
mesh_df.head(2)

mesh_df.to_csv('data/mesh-intervention-to-condition.tsv', sep='\t', index=False)

