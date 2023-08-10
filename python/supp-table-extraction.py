get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import sys
import cPickle
import numpy as np
import sqlalchemy

# set the paths to snorkel and gwasdb
sys.path.append('../snorkel-tables')
sys.path.append('../src')
sys.path.append('../src/crawler')

# set up the directory with the input papers
supp_root_dir = '/Users/kuleshov/Downloads/gwas_supp2/'
supp_sub_dirs = ['doc/', 'docx/', 'xls/', 'xlsx/']
supp_dirs = [supp_root_dir + d for d in supp_sub_dirs]
supp_map_file = '../data/db/supp-map.txt'

# set up matplotlib
import matplotlib
get_ipython().magic('matplotlib inline')
matplotlib.rcParams['figure.figsize'] = (12,4)

# create a Snorkel session
from snorkel import SnorkelSession
session = SnorkelSession()

from extractor.parser import SuppXMLDocParser

xml_parser = SuppXMLDocParser(
    paths=supp_dirs,
    map_path=supp_map_file,
    doc='.//body',
    text='.//table',
    keep_xml_tree=True)

session.rollback()

from snorkel.models import Corpus
from snorkel.parser import CorpusParser, OmniParser

# parses tables into rows, cols, cells...
table_parser = OmniParser(timeout=1000000)

try:
    corpus = session.query(Corpus).filter(Corpus.name == 'GWAS Supplementary Table Corpus 13').one()
except:
    cp = CorpusParser(xml_parser, table_parser, max_docs=3)
    get_ipython().magic("time corpus = cp.parse_corpus(name='GWAS Supplementary Table Corpus 13', session=session)")
    session.add(corpus)
    session.commit()

print 'Loaded corpus of %d documents' % len(corpus)

for d in corpus.documents:
    if d.tables: 
        for c in d.tables[0].cells[:50]:
            print c.text
            print c.phrases
#     if '1471-2261-11-29-s12' in d.name:
#         print d, d.meta['root'][:100]

print corpus.documents[0].name

import lxml.etree as et

d=et.parse('/Users/kuleshov/Downloads/gwas_supp2/doc/1471-2261-11-29-s12.html', et.HTMLParser()).xpath('.//table')

from snorkel.matchers import RegexMatchSpan
rsid_matcher = RegexMatchSpan(rgx=r'rs\d+(/[ATCG]{1,2})*$')

from snorkel.candidates import TableNgrams
from snorkel.matchers import RegexMatchSpan, Union

# 1: p-value matcher

rgx1 = u'[1-9]\d?[\xb7\.]?\d*[\s\u2009]*[\xd7\xb7\*][\s\u2009]*10[\s\u2009]*[-\u2212\u2013\u2012][\s\u2009]*\d+'
pval_rgx_matcher1 = RegexMatchSpan(rgx=rgx1)
rgx2 = u'[1-9]\d?[\xb7\.]?\d*[\s\u2009]*[eE][\s\u2009]*[-\u2212\u2013\u2012][\s\u2009]*\d+'
pval_rgx_matcher2 = RegexMatchSpan(rgx=rgx2)
rgx3 = u'0\.0000+\d+'
pval_rgx_matcher3 = RegexMatchSpan(rgx=rgx3)
pval_rgx_matcher = Union(pval_rgx_matcher1, pval_rgx_matcher2, pval_rgx_matcher3)

# 2: column-based matcher (currently not used)

from snorkel.matchers import CellNameRegexMatcher

pval_rgx = 'p\s?.?\s?value'
pval_rgxname_matcher = CellNameRegexMatcher(axis='col', rgx=pval_rgx, n_max=3, ignore_case=True, header_only=True, max_chars=20)

# 3: combine the two

pval_matcher = Union(pval_rgx_matcher, pval_rgxname_matcher)

# create a Snorkel class for the relation we will extract
from snorkel.models import candidate_subclass
RsidPhenRel = candidate_subclass('RsidPvalRel', ['rsid','pval'])

# define our candidate spaces
from snorkel.candidates import TableNgrams
unigrams = TableNgrams(n_max=1)
heptagrams = TableNgrams(n_max=7)

# we will be looking only at aligned cells
from snorkel.throttlers import AlignmentThrottler
row_align_filter = AlignmentThrottler(axis='row', infer=False)

# the first extractor looks at phenotype names in columns with a header indicating it's a phenotype
from snorkel.candidates import CandidateExtractor
ce = CandidateExtractor(RsidPhenRel, [unigrams, heptagrams], [rsid_matcher, pval_rgx_matcher], throttler=row_align_filter)

# collect that cells that will be searched for candidates
tables = [table for doc in corpus.documents for table in doc.tables]

from snorkel.models import CandidateSet
    
try:
    rels = session.query(CandidateSet).filter(CandidateSet.name == 'RsidPvalRel Relations 2').one()
except:
    get_ipython().magic("time rels = ce.extract(tables, 'RsidPvalRel Relations 2', session)")
    session.add(rels)
    session.commit()

print "%s relations extracted, e.g." % len(rels)
for cand in rels[:10]:
    print cand

import re
from extractor.util import pvalue_to_float

def clean_rsid(rsid):
    return re.sub('/.+', '', rsid)

with open('results/nb-output/supp-pval-rsid.tsv', 'w') as f:
    for rel in rels:
        docname = rel[0].parent.document.name
        pmid = docname.split('-')[0]
        table_id = rel[0].parent.table.position
        row_id = rel[0].parent.cell.row.position
        col_id = rel[0].parent.cell.col.position
        rsid = rel[0].get_span()
        log_pval = pvalue_to_float(rel[1].get_span())
        
        try:
            out_str = '%s\t%s\t%d\t%d\t%d\t%f\n' % (pmid, clean_rsid(rsid), table_id, row_id, col_id, log_pval)
        except:
            print 'could not write:', pmid, clean_rsid(rsid), table_id, row_id, col_id, log_pval
        f.write(out_str)

# Define the extractor
from snorkel.models import candidate_subclass
from snorkel.matchers import RegexMatchSpan
from snorkel.candidates import CandidateExtractor

RSID = candidate_subclass('SnorkelRsid 2', ['rsid'])

unigrams = TableNgrams(n_max=1)
rsid_singleton_matcher = RegexMatchSpan(rgx=r'rs\d+(/[^s]+)?')
rsid_singleton_extractor = CandidateExtractor(RSID, unigrams, rsid_singleton_matcher)

from snorkel.models import CandidateSet

try:
    rsid_c = session.query(CandidateSet).filter(CandidateSet.name == 'Rsid Candidates 2').one()
except:
    tables = [table for doc in corpus.documents for table in doc.tables]
    print '%d tables loaded' % len(tables)
    get_ipython().magic("time rsid_c = rsid_singleton_extractor.extract(tables, 'Rsid Candidates 2', session)")
    session.add(rsid_c)
    session.commit()

print '%d candidates extracted' % len(rsid_c)

rsid_by_table = dict()
for cand in rsid_c:
    rsid = cand[0].get_span()
    key = cand[0].parent.document.name, cand[0].parent.table.position
    if key not in rsid_by_table: rsid_by_table[key] = set()
    rsid_by_table[key].add((rsid, cand[0].parent.cell.row.position, cand[0].parent.cell.col.position))
    
with open('results/nb-output/rsids.singletons.all.tsv', 'w') as f:
    for (pmid, table_id), rsids in rsid_by_table.items():
        if len(rsids) < 10: continue
        for rsid, row_num, col_num in rsids:
            f.write('%s\t%s\t%s\t%s\t%s\n' % (pmid, table_id, row_num, col_num, rsid))

import re
from bs4 import BeautifulSoup as soup

pval_rgx = 'p\s?.?\s?value'
lod_rgx = 'LOD'

def has_pval(txt):
    if re.search(pval_rgx, txt, re.IGNORECASE):
        return True
    elif txt.lower() == 'p':
        return True
    return False

with open('results/nb-output/table-annotations.tsv', 'w') as f:
    for doc in corpus.documents:
        for table in doc.tables:
            lod_found = 0
            pval_found = 0
            for cell in table.cells:
                txt = soup(cell.text).text
                if not pval_found and len(txt) < 30 and has_pval(txt):
                    pval_found = 1
                if not lod_found and re.search(lod_rgx, txt):
                    lod_found = 1
                if pval_found and lod_found: break
                    
            out_str = '%s\t%s\t%s\t%s\n' % (doc.name, table.position, pval_found, lod_found)
            f.write(out_str) 

rels = []
loc2rsid = dict()
with open('results/nb-output/supp-pval-rsid.tsv') as f:
    for line in f:
        pmid, rsid, table_id, row_id, col_id, pval = line.strip().split('\t')
        loc = pmid, table_id, row_id
        rels.append((pmid, rsid, table_id, row_id, col_id, pval))
        if loc not in loc2rsid: loc2rsid[loc] = set()
        loc2rsid[loc].add(rsid)

n = 0
with open('results/nb-output/supp-pval-rsid.filtered.tsv', 'w') as f:
    for rel in rels:
        pmid, rsid, table_id, row_id, col_id, pval = rel
        loc = pmid, table_id, row_id
        if len(loc2rsid[loc]) > 1: continue
        
        out_str = '%s\t%s\t%s\t%s\t%s\t%s\n' % (pmid, rsid, table_id, row_id, col_id, pval)
        f.write(out_str)
        n += 1
        
print len(rels), n



