import logging

from collections import Counter
from operator import itemgetter

from lxml import etree

from rdkit import Chem
from rdkit.Chem import Draw 
from rdkit.Chem.Draw import IPythonConsole

from IPython.display import Image, display

# Python modules used for API access...
# By default, the API connects to the main ChEMBL database; set it to use the local version (i.e. myChEMBL) instead...
from chembl_webresource_client.settings import Settings
Settings.Instance().NEW_CLIENT_URL = 'http://localhost/chemblws'

from chembl_webresource_client.new_client import new_client

available_resources = [resource for resource in dir(new_client) if not resource.startswith('_')]
print available_resources
print len(available_resources)

# Get a molecule-handler object for API access and check the connection to the database...

molecule = new_client.molecule
molecule.set_format('json')
print "%s molecules available in myChEMBL_20" % len(molecule.all())

# so this:
# 1.
m1 = molecule.get('CHEMBL25')
# 2.
m2 = molecule.get('BSYNRYMUTXBXSQ-UHFFFAOYSA-N')
#
m3 = molecule.get('CC(=O)Oc1ccccc1C(=O)O')
# will return the same data:
m1 == m2 == m3

# Lapatinib, the bioactive component of the anti-cancer drug Tykerb

chembl_id = "CHEMBL554" 

# Get compound record using client...

record_via_client = molecule.get(chembl_id)

record_via_client

# Import a Python module to allow URL-based access...

import requests
from urllib import quote

# Stem of URL for local version of web services...

url_stem = "http://localhost/chemblws"

# Note that, for historical reasons, the URL-based webservices return XML by default, so JSON
# must be requested explicity by appending '.json' to the URL.

# Get request object...
url = url_stem + "/molecule/" + chembl_id + ".json"
request = requests.get(url)

print url

# Check reqest status: should be 200 if everything went OK...
print request.status_code

record_via_url = request.json()
record_via_url 

record_via_client == record_via_url

smiles_from_json = record_via_client['molecule_structures']['canonical_smiles']

# Get compound record in XML format...

molecule.set_format('xml')
xml = molecule.get(chembl_id).encode('utf-8')
#print xml
# The XML must be parsed (e.g. using the lxml.etree module in Python) to enable extraction of the data...

root = etree.fromstring(xml).getroottree()

# Extract SMILES via xpath...

smiles_from_xml = root.xpath("/molecule/molecule_structures/canonical_smiles/text()")[0]

print smiles_from_xml
print smiles_from_xml == smiles_from_json

# Pretty-print XML...

print etree.tostring(root, pretty_print=True)

# InChI Key for Lapatinib
inchi_key = "BCFGMOOMADDAQU-UHFFFAOYSA-N"

# getting molecule via client
molecule.set_format('json')
record_via_client = molecule.get(inchi_key)

# getting molecule via url
url = url_stem + "/molecule/" + inchi_key + ".json"
record_via_url = requests.get(url).json()

print url

# they are the same
print record_via_url == record_via_client

# Canonoical SMILES for Lapatinib
canonical_smiles = "CS(=O)(=O)CCNCc1oc(cc1)c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2"

# getting molecule via client
molecule.set_format('json')
record_via_client = molecule.get(canonical_smiles)

# getting molecule via url
url = url_stem + "/molecule/" + quote(canonical_smiles) + ".json"
record_via_url = requests.get(url).json()

print url

# they are the same
record_via_url == record_via_client

records1 = molecule.get(['CHEMBL6498', 'CHEMBL6499', 'CHEMBL6505'])
records2 = molecule.get(['XSQLHVPPXBBUPP-UHFFFAOYSA-N', 'JXHVRXRRSSBGPY-UHFFFAOYSA-N', 'TUHYVXGNMOGVMR-GASGPIRDSA-N'])
records3 = molecule.get(['CNC(=O)c1ccc(cc1)N(CC#C)Cc2ccc3nc(C)nc(O)c3c2',
            'Cc1cc2SC(C)(C)CC(C)(C)c2cc1\\N=C(/S)\\Nc3ccc(cc3)S(=O)(=O)N',
            'CC(C)C[C@H](NC(=O)[C@@H](NC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)[C@H]3CCCN3C(=O)C(CCCCN)CCCCN)C(C)(C)C)C(=O)O'])
records1 == records2 == records3

url1 = url_stem + "/molecule/set/%s;%s;%s" % ('CHEMBL6498', 'CHEMBL6499', 'CHEMBL6505') + ".json"
records1 = requests.get(url1).json()

url2 = url_stem + "/molecule/set/%s;%s;%s" % ('XSQLHVPPXBBUPP-UHFFFAOYSA-N', 'JXHVRXRRSSBGPY-UHFFFAOYSA-N', 'TUHYVXGNMOGVMR-GASGPIRDSA-N') + ".json"
records2 = requests.get(url2).json()

url3 = url_stem + "/molecule/set/%s;%s;%s" % (quote('CNC(=O)c1ccc(cc1)N(CC#C)Cc2ccc3nc(C)nc(O)c3c2'),
            quote('Cc1cc2SC(C)(C)CC(C)(C)c2cc1\\N=C(/S)\\Nc3ccc(cc3)S(=O)(=O)N'),
            quote('CC(C)C[C@H](NC(=O)[C@@H](NC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)[C@H]3CCCN3C(=O)C(CCCCN)CCCCN)C(C)(C)C)C(=O)O')) + ".json"
records3 = requests.get(url3).json()

print url1
print url2
print url3

records1 == records2 == records3

# Generate a list of 300 ChEMBL IDs (N.B. not all will be valid)...

chembl_ids = ['CHEMBL{}'.format(x) for x in range(1, 301)]

# Get compound records, note `molecule_chembl_id` named parameter.
# Named parameters should always be used for longer lists

records = molecule.get(molecule_chembl_id=chembl_ids)

len(records)

# First, filtering using the client:

# 1. Get all approved drugs
approved_drugs = molecule.filter(max_phase=4)

# 2. Get all molecules in ChEMBL with no Rule-of-Five violations
no_violations = molecule.filter(molecule_properties__num_ro5_violations=0)

# 3. Get all biotherapeutic molecules
biotherapeutics = molecule.filter(biotherapeutic__isnull=False)

# 4. Return molecules with molecular weight <= 300
light_molecules = molecule.filter(molecule_properties__mw_freebase__lte=300)

# 5. Return molecules with molecular weight <= 300 AND pref_name ends with nib
light_nib_molecules = molecule.filter(molecule_properties__mw_freebase__lte=300).filter(pref_name__iendswith="nib")

# Secondly, fltering using url endpoint:

# 1. Get all approved drugs
url_1 = url_stem + "/molecule.json?max_phase=4"
url_approved_drugs = requests.get(url_1).json()

# 2. Get all molecules in ChEMBL with no Rule-of-Five violations
url_2 = url_stem + "/molecule.json?molecule_properties__num_ro5_violations=0"
ulr_no_violations = requests.get(url_2).json()

# 3. Get all biotherapeutic molecules
url_3 = url_stem + "/molecule.json?biotherapeutic__isnull=false"
url_biotherapeutics = requests.get(url_3).json()

# 4. Return molecules with molecular weight <= 300
url_4 = url_stem + "/molecule.json?molecule_properties__mw_freebase__lte=300"
url_light_molecules = requests.get(url_4).json()

# 5. Return molecules with molecular weight <= 300 AND pref_name ends with nib
url_5 = url_stem + "/molecule.json?molecule_properties__mw_freebase__lte=300&pref_name__iendswith=nib"
url_light_nib_molecules = requests.get(url_5).json()

print url_1
print url_2
print url_3
print url_4
print url_5

# First off, they are not the same thing:
print approved_drugs == url_approved_drugs

# Not surprisingly, url-endpoint produced JSON data, which has been paresed into python dict:
print type(url_approved_drugs)

# Whereas the client has returned an object of type `QuerySet`
print type(approved_drugs)

# Let's examine what data contains the python dict:
url_approved_drugs

# The default size of single page is 20 results:
len(url_approved_drugs['molecules'])

# But it can be extended up to 1000 results by providing `limit` argument:
url = url_stem + "/molecule.json?max_phase=4&limit=200"
bigger_page = requests.get(url).json()

print url
print len(bigger_page['molecules'])

#Let's see what data is provided in `page-meta` dictionary:
url_approved_drugs['page_meta']

# Getting all approved drugs using url endpoint
localhost = "http://localhost/"
url_approved_drugs = requests.get(localhost + "chemblws/molecule.json?max_phase=4&limit=1000").json()
results = url_approved_drugs['molecules']
while url_approved_drugs['page_meta']['next']:
    url_approved_drugs = requests.get(localhost + url_approved_drugs['page_meta']['next']).json()
    results += url_approved_drugs['molecules']
print len(results)
print len(results) == url_approved_drugs['page_meta']['total_count']

# The QuerySet object returned by the client is a lazily-evaluated iterator
# This means that it's ready to use and it will try to reduce the amount of server requests
# All results are cached as well so they are fetched from server only once.
approved_drugs = molecule.filter(max_phase=4)

# Getting the lenght of the whole result set is easy:
print len(approved_drugs)

# So is getting a single element:
print approved_drugs[123]

# Or a chunk of elements:
print approved_drugs[2:5]

# Or using in the loops or list comprehensions:
drug_smiles = [drug['molecule_structures']['canonical_smiles'] for drug in approved_drugs if drug['molecule_structures']]
print len(drug_smiles)

# Sort approved drugs by molecular weight ascending (from lightest to heaviest) and get the first (lightest) element
lightest_drug = molecule.filter(max_phase=4).order_by('molecule_properties__mw_freebase')[0]
lightest_drug['pref_name']

# Sort approved drugs by molecular weight descending (from heaviest to lightest) and get the first (heaviest) element
heaviest_drug = molecule.filter(max_phase=4).order_by('-molecule_properties__mw_freebase')[0]
heaviest_drug['pref_name']

# Do the same using url endpoint
url_1 = url_stem + "/molecule.json?max_phase=4&order_by=molecule_properties__mw_freebase"
lightest_drug = requests.get(url_1).json()['molecules'][0]
print url_1
print lightest_drug['pref_name']

url_2 = url_stem + "/molecule.json?max_phase=4&order_by=-molecule_properties__mw_freebase"
heaviest_drug = requests.get(url_2).json()['molecules'][0]
print url_2
print heaviest_drug['pref_name']

# Atorvastatin...
smiles = "CC(C)c1c(C(=O)Nc2ccccc2)c(c3ccccc3)c(c4ccc(F)cc4)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"

# By default, the type of search used is 'exact search' which means that only compounds with exacly same SMILES string will be picked:
result = molecule.filter(molecule_structures__canonical_smiles=smiles)
print len(result)

# This is quivalent of:
result1 = molecule.filter(molecule_structures__canonical_smiles__exact=smiles)
print len(result1)

# For convenience, we have a shortcut call:
result2 = molecule.filter(smiles=smiles)
print len(result2)

# Checking if they are all the same: 
print result[0]['pref_name'] == result1[0]['pref_name'] == result2[0]['pref_name']

# And because SMILES string are unique in ChEMBL, this is similar to:
result3 = molecule.get(smiles)
print result[0]['pref_name'] == result3['pref_name']

# Flexmatch will look for structures that match given SMILES, ignoring stereo:
records = molecule.filter(molecule_structures__canonical_smiles__flexmatch=smiles)
print len(records)

for record in records:
    print("{:15s} : {}".format(record["molecule_chembl_id"], record['molecule_structures']['canonical_smiles']))

# The same can be achieved using url endpoint:

url_1 = url_stem + "/molecule.json?molecule_structures__canonical_smiles=" + quote(smiles)
url_2 = url_stem + "/molecule.json?molecule_structures__canonical_smiles__exact=" + quote(smiles)
url_3 = url_stem + "/molecule.json?smiles=" + quote(smiles)
url_4 = url_stem + "/molecule.json?molecule_structures__canonical_smiles__flexmatch=" + quote(smiles)

exact_match = requests.get(url_1).json()
explicit_exact_match = requests.get(url_2).json()
convenient_shortcut = requests.get(url_3).json()
flexmatch = requests.get(url_4).json()

print url_1
print len(exact_match['molecules'])

print url_2
print len(explicit_exact_match['molecules'])

print url_3
print len(convenient_shortcut['molecules'])

print url_4
print len(flexmatch['molecules'])

print exact_match == explicit_exact_match

# CHEMBL477889
smiles = "[Na+].CO[C@@H](CCC#C\C=C/CCCC(C)CCCCC=C)C(=O)[O-]"

url = url_stem + "/molecule/" + smiles + ".json"
result = requests.get(url)

print url
print result.ok
print result.status_code

# Method one:
url = url_stem + "/molecule/" + quote(smiles) + ".json"
result_by_get = requests.get(url)

print url
print result_by_get.ok
print result_by_get.status_code

# Method two:
url = url_stem + "/molecule.json"
result_by_post = requests.post(url, data={"smiles": smiles}, headers={"X-HTTP-Method-Override": "GET"})

print result_by_post.ok
print result_by_post.status_code

print smiles
print result_by_post.json()
print result_by_get.json() == result_by_post.json()['molecules'][0]

# Lapatinib contains the following core...

query = "c4ccc(Nc2ncnc3ccc(c1ccco1)cc23)cc4"

Chem.MolFromSmiles(query)

# Perform substructure search on query using client

substructure = new_client.substructure
records = substructure.filter(smiles=query)

# ... and using raw url-endpoint

url = url_stem + "/substructure/" + quote(query) + ".json"
result = requests.get(url).json()

print url
print result['page_meta']['total_count']

mols = [Chem.MolFromSmiles(x['molecule_structures']['canonical_smiles']) for x in records[:6]]
legends=[str(x["molecule_chembl_id"]) for x in records]
Draw.MolsToGridImage(mols, legends=legends, subImgSize=(400, 200), useSVG=False)

# Lapatinib
smiles = "CS(=O)(=O)CCNCc1oc(cc1)c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2"

# Peform similarity search on molecule using client...

# Note that a percentage similarity must be supplied.
similarity = new_client.similarity
res = similarity.filter(smiles=smiles, similarity=85)

len(res)

##### ... and using raw url-endpoint

url = url_stem + "/similarity/" + quote(smiles) + "/85.json"
result = requests.get(url).json()

print url
print result['page_meta']['total_count']

mols = [Chem.MolFromSmiles(x['molecule_structures']['canonical_smiles']) for x in res[:6]]
legends = [str(x["molecule_chembl_id"]) for x in res]
Draw.MolsToGridImage(mols, legends=legends, subImgSize=(400, 200), useSVG=False)

# Neostigmine (a parent)...

chembl_id = "CHEMBL278020" 

records = new_client.molecule_form.get(chembl_id)['molecule_forms']

records

for chembl_id in [x["molecule_chembl_id"] for x in records if x["parent"] == 'False']:
    record = new_client.molecule.get(chembl_id)          
    print("{:10s} : {}".format(chembl_id, record['molecule_structures']['canonical_smiles']))

# Molecule forms for Lapatinib are used here...

for chembl_id in (x["molecule_chembl_id"] for x in new_client.molecule_form.get("CHEMBL554")['molecule_forms']):
        
    print("The recorded mechanisms of action of '{}' are...".format(chembl_id))
        
    mechanism_records = new_client.mechanism.filter(molecule_chembl_id=chembl_id)
    
    if mechanism_records:
    
        for mech_rec in mechanism_records:
    
            print("{:10s} : {}".format(mech_rec["molecule_chembl_id"], mech_rec["mechanism_of_action"]))
        
    print("-" * 50)

# Lapatinib ditosylate monohydrate (Tykerb)

chembl_id = "CHEMBL1201179" 

png = new_client.image.get(chembl_id)

Image(png)

# Lapatinib

chembl_id = "CHEMBL554" 

records = new_client.activity.filter(molecule_chembl_id=chembl_id)

len(records), records[:2]

# Like with any other resource type, a complete list of targets can be requested using the client:
records = new_client.target.all()
len(records)

records[:4]

# Count target types...

counts = Counter([x["target_type"] for x in records if x["target_type"]])

for targetType, n in sorted(counts.items(), key=itemgetter(1), reverse=True): print("{:30s} {:-4d}".format(targetType, n))

# Receptor protein-tyrosine kinase erbB-2
    
chembl_id = "CHEMBL1824"

record = new_client.target.get(chembl_id)

record

# SK-BR-3, a cell line over-expressing erbB-2

chembl_id = "CHEMBL613834" 

record = new_client.target.get(chembl_id)

record

# UniProt ID for erbB-2, a target of Lapatinib

uniprot_id = "P04626"

records = new_client.target.filter(target_components__accession=uniprot_id)
print [(x['target_chembl_id'], x['pref_name']) for x in records]

# Receptor protein-tyrosine kinase erbB-2

chembl_id = "CHEMBL1824"

records = new_client.activity.filter(target_chembl_id=chembl_id)

len(records)

# Show assays with most recorded bioactivities...

for assay, n in sorted(Counter((x["assay_chembl_id"], x["assay_description"]) for x in records).items(), key=itemgetter(1), reverse=True)[:5]:
    
    print("{:-4d} {:14s} {}".format(n, *assay))

# Receptor protein-tyrosine kinase erbB-2

chembl_id = "CHEMBL1824"

activities = new_client.mechanism.filter(target_chembl_id=chembl_id)
compound_ids = [x['molecule_chembl_id'] for x in activities]
approved_drugs = new_client.molecule.filter(molecule_chembl_id__in=compound_ids).filter(max_phase=4)

for record in approved_drugs:
    
    print("{:10s} : {}".format(record["molecule_chembl_id"], record["pref_name"]))

# Inhibitory activity against epidermal growth factor receptor

chembl_id = "CHEMBL674106"

record = new_client.assay.get(chembl_id)

record

records = new_client.activity.filter(assay_chembl_id=chembl_id)

len(records), records[:2]

# Documents - retrieve all publications published after 1985 in 5th volume.
print new_client.document.filter(doc_type='PUBLICATION').filter(year__gt=1985).filter(volume=5)

# Cell lines:
print new_client.cell_line.get('CHEMBL3307242')

# Protein class:
print new_client.protein_class.filter(l6="CAMK protein kinase AMPK subfamily")

# Source:
print new_client.source.filter(src_short_name="ATLAS")

# Target component:
print new_client.target_component.get(375)

# ChEMBL ID Lookup: check if CHEMBL1 is a molecule, assay or target:
print new_client.chembl_id_lookup.get("CHEMBL1")['entity_type']

# ATC class:
print new_client.atc_class.get('H03AA03')

