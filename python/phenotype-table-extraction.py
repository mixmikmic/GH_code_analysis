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
abstract_dir = '../data/db/papers'

# set up matplotlib
import matplotlib
get_ipython().magic('matplotlib inline')
matplotlib.rcParams['figure.figsize'] = (12,4)

# create a Snorkel session
from snorkel import SnorkelSession
session = SnorkelSession()

from extractor.parser import UnicodeXMLTableDocParser
from snorkel.parser import XMLMultiDocParser

xml_parser = XMLMultiDocParser(
    path=abstract_dir,
    doc='./*',
    text='.//table',
    id='.//article-id[@pub-id-type="pmid"]/text()',
    keep_xml_tree=True)

from snorkel.parser import CorpusParser, OmniParser
from snorkel.models import Corpus

# parses tables into rows, cols, cells...
table_parser = OmniParser(timeout=1000000)

try:
    corpus = session.query(Corpus).filter(Corpus.name == 'GWAS Table Corpus').one()
except:
    cp = CorpusParser(xml_parser, table_parser)
    get_ipython().magic("time corpus = cp.parse_corpus(name='GWAS Table Corpus', session=session)")
    session.add(corpus)
    session.commit()

print 'Loaded corpus of %d documents' % len(corpus)

from snorkel.matchers import RegexMatchSpan
rsid_matcher = RegexMatchSpan(rgx=r'rs\d+(/[ATCG]{1,2})*$')

from snorkel.matchers import CellNameDictionaryMatcher

phen_words = ['trait', 'phenotype', 'outcome'] # words that denote phenotypes
phen_matcher = CellNameDictionaryMatcher(axis='col', d=phen_words, n_max=3, ignore_case=True)

from snorkel.matchers import DictionaryMatch
from db.kb import KnowledgeBase
from extractor.util import make_ngrams

# collect phenotype list
kb = KnowledgeBase()
# efo phenotypes
efo_phenotype_list0 = kb.get_phenotype_candidates(source='efo', peek=True) # TODO: remove peaking
efo_phenotype_list = list(make_ngrams(efo_phenotype_list0))
# mesh diseases
mesh_phenotype_list0 = kb.get_phenotype_candidates(source='mesh')
mesh_phenotype_list = list(make_ngrams(mesh_phenotype_list0))
# mesh chemicals
chem_phenotype_list = kb.get_phenotype_candidates(source='chemical')

phenotype_names = efo_phenotype_list + mesh_phenotype_list + chem_phenotype_list
phen_name_matcher = DictionaryMatch(d=phenotype_names, ignore_case=True, stemmer='porter')

from snorkel.candidates import CandidateExtractor
from snorkel.throttlers import AlignmentThrottler, SeparatingSpanThrottler, OrderingThrottler, CombinedThrottler

# create a Snorkel class for the relation we will extract
from snorkel.models import candidate_subclass
RsidPhenRel = candidate_subclass('RsidPhenRel', ['rsid','phen'])

# define our candidate spaces
from snorkel.candidates import TableNgrams, TableCells, SpanningTableCells
unigrams = TableNgrams(n_max=1)
cells = TableCells()
spanning_cells = SpanningTableCells(axis='row')

# we will be looking only at aligned cells
row_align_filter = AlignmentThrottler(axis='row', infer=True)

# and at cells where the phenotype is in a spanning header cell above the rsid cell
sep_span_filter = SeparatingSpanThrottler(align_axis='col') # rsid and phen are not separated by spanning cells
col_order_filter = OrderingThrottler(axis='col', first=1) # phen spanning cell comes first
header_filter = CombinedThrottler([sep_span_filter, col_order_filter]) # combine the two throttlers

# the first extractor looks at phenotype names in columns with a header indicating it's a phenotype
ce1 = CandidateExtractor(RsidPhenRel, [unigrams, cells], [rsid_matcher, phen_matcher], throttler=row_align_filter)

# the second extractor looks at phenotype names in columns with a header indicating it's a phenotype
ce2 = CandidateExtractor(RsidPhenRel, [unigrams, spanning_cells], [rsid_matcher, phen_name_matcher], throttler=header_filter, stop_on_duplicates=False)

# collect that cells that will be searched for candidates
tables = [table for doc in corpus.documents for table in doc.tables]

from snorkel.models import CandidateSet

try:
    rels1 = session.query(CandidateSet).filter(CandidateSet.name == 'RsidPhenRel Set 1').one()
except:
    get_ipython().magic("time rels1 = ce1.extract(tables, 'RsidPhenRel Set 1', session)")
    
print "%s relations extracted, e.g." % len(rels1)
for cand in rels1[:10]:
    print cand

from snorkel.models import CandidateSet

try:
    rels2 = session.query(CandidateSet).filter(CandidateSet.name == 'RsidPhenRel Set 2').one()
except:
    get_ipython().magic("time rels2 = ce2.extract(tables, 'RsidPhenRel Set 2', session)")
    
print "%s relations extracted, e.g." % len(rels2)
for cand in rels2[:10]: 
    print cand

from snorkel.models import CandidateSet

try:
    rels = session.query(CandidateSet).filter(CandidateSet.name == 'RsidPhenRel Canidates').one()
except:
    rels = CandidateSet(name='RsidPhenRel Canidates')
    for c in rels1: rels.append(c)
    for c in rels2: rels.append(c)

    session.add(rels)
    session.commit()

print '%d candidates in total' % len(rels)

try:
    train_c = session.query(CandidateSet).filter(CandidateSet.name == 'RsidPhenRel Training Candidates').one()
    devtest_c = session.query(CandidateSet).filter(CandidateSet.name == 'RsidPhenRel Dev/Test Candidates').one()
except:
    # delete any previous sets with that name
    session.query(CandidateSet).filter(CandidateSet.name == 'RsidPhenRel Training Candidates').delete()
    session.query(CandidateSet).filter(CandidateSet.name == 'RsidPhenRel Dev/Test Candidates').delete()

    frac_test = 0.5

    # initialize the new sets
    train_c = CandidateSet(name='RsidPhenRel Training Candidates')
    devtest_c = CandidateSet(name='RsidPhenRel Dev/Test Candidates')

    # choose a random subset for the labeled set
    n_test = len(rels) * frac_test
    test_idx = set(np.random.choice(len(rels), size=(n_test,), replace=False))

    # add to the sets
    for i, c in enumerate(rels):
        if i in test_idx:
            devtest_c.append(c)
        else:
            train_c.append(c)

    # save the results
    session.add(train_c)
    session.add(devtest_c)
    session.commit()

print 'Initialized %d training and %d dev/testing candidates' % (len(train_c), len(devtest_c))

from snorkel.lf_helpers import *
s=None
doc = [d for d in corpus.documents if d.name == '17903303'][0]
table = doc.tables[3]
for cell in table.cells:
    top_cells = get_aligned_cells(cell, 'col', infer=True)
    top_phrases = [phrase for cell in top_cells for phrase in cell.phrases]
# rels[0][1].parent.table.cells[0].phrases
# corpus.documents[0].phrases

from snorkel.lf_helpers import *

bad_words = ['rs number', 'rs id', 'rsid']

# negative LFs
def LF_number(m):
    txt = m[1].get_span()
    frac_num = len([ch for ch in txt if ch.isdigit()]) / float(len(txt))
    return -1 if len(txt) > 5 and frac_num > 0.4 or frac_num > 0.6 else 0

def LF_bad_phen_mentions(m):
    if cell_spans(m[1].parent.cell, m[1].parent.table, 'row'): return 0
    #     if m[1].context.cell.spans('row'): return 0
    top_cells = get_aligned_cells(m[1].parent.cell, 'col', infer=True)
    top_cells = [cell for cell in top_cells]
#     top_cells = m.span1.context.cell.aligned_cells(axis='col', induced=True)
    try:
        top_phrases = [phrase for cell in top_cells for phrase in cell.phrases]
    except:
        for cell in top_cells:
            print cell, cell.phrases
    if not top_phrases: return 0
    matching_phrases = []
    for phrase in top_phrases:
        if any (phen_matcher._f_ngram(word) for word in phrase.text.split(' ')):
            matching_phrases.append(phrase)
    small_matching_phrases = [phrase for phrase in matching_phrases if len(phrase.text) <= 25]
    return -1 if not small_matching_phrases else 0

def LF_bad_word(m):
    txt = m[1].get_span()
    return -1 if any(word in txt for word in bad_words) else 0

LF_tables_neg = [LF_number, LF_bad_phen_mentions]

# positive LFs
def LF_no_neg(m):
    return +1 if not any(LF(m) for LF in LF_tables_neg) else 0

LF_tables_pos = [LF_no_neg]

LFs = LF_tables_neg + LF_tables_pos

from snorkel.annotations import LabelManager
label_manager = LabelManager()

try:
    get_ipython().magic("time L_train = label_manager.load(session, train_c, 'RsidPhenRel LF Labels6')")
except sqlalchemy.orm.exc.NoResultFound:
    get_ipython().magic("time L_train = label_manager.create(session, train_c, 'RsidPhenRel LF Labels6', f=LFs)")

L_train.lf_stats()

from snorkel.learning import NaiveBayes

gen_model = NaiveBayes()
gen_model.train(L_train, n_iter=10000, rate=1e-2)

gen_model.w

from snorkel.annotations import LabelManager
label_manager = LabelManager()

# delete existing labels
# session.rollback()
# session.query(AnnotationKeySet).filter(AnnotationKeySet.name == 'RsidPhenRel LF All Labels').delete()
get_ipython().magic("time L_all = label_manager.create(session, rels, 'RsidPhenRel LF All Lab', f=LFs)")

preds = gen_model.odds(L_all)
good_rels = [(c[0].parent.document.name, c[0].get_span(), c[1].get_span()) for (c, p) in zip(rels, preds) if p > 0]
print len(good_rels), 'relations extracted, e.g.:'
print good_rels[:10]

# store relations to annotate
with open('results/nb-output/rels.acronyms.extracted.tsv', 'w') as f:
    for doc_id, str1, str2 in good_rels:
        try:
            out = u'{}\t{}\t{}\n'.format(doc_id, unicode(str1), str2)
            f.write(out.encode("UTF-8"))
        except:
            print 'Error saving:', str1, str2

from extractor.dictionary import Dictionary, unravel

D = Dictionary()
D.load('results/nb-output/acronyms.extracted.all.tsv')
print len(D), 'definitions loaded'

pval_rsid_dict = dict()
pval_dict = dict() # combine all of the pvalues for a SNPs in the same document into one set
with open('results/nb-output/pval-rsid.tsv') as f:
    for line in f:
        pmid, rsid, table_id, row_id, col_id, log_pval = line.strip().split('\t')
        log_pval, table_id, row_id, col_id = float(log_pval), int(table_id), int(row_id), int(col_id)
        
        if pmid not in pval_rsid_dict: pval_rsid_dict[pmid] = dict()
        key = (rsid, table_id, row_id)
        if key not in pval_rsid_dict[pmid]: pval_rsid_dict[pmid][key] = set()
        pval_rsid_dict[pmid][key].add(log_pval)
                
        if pmid not in pval_dict: pval_dict[pmid] = dict()
        if rsid not in pval_dict[pmid]: pval_dict[pmid][rsid] = set()
        pval_dict[pmid][rsid].add(log_pval)

pval_dict0 = {pmid : {rsid : min(pval_dict[pmid][rsid]) for rsid in pval_dict[pmid]} for pmid in pval_dict}
pval_rsid_dict0 = {pmid : {key : min(pval_rsid_dict[pmid][key]) for key in pval_rsid_dict[pmid]} for pmid in pval_rsid_dict}
pval_dict = pval_dict0
pval_rsid_dict = pval_rsid_dict0

# preds = learner.predict_wmv(candidates)
predicted_candidates =  [c for (c, p) in zip(rels, preds) if p > 0]

import re
import unicodedata
def _normalize_str(s):
    try:
        s = s.encode('utf-8')
        return s
    except UnicodeEncodeError: 
        pass
    try:
        s = s.decode('utf-8')
        return s
    except UnicodeDecodeError: 
        pass    
    raise Exception()
    
def clean_rsid(rsid):
    return re.sub('/.+', '', rsid)

with open('results/nb-output/phen-rsid.table.rel.all.tsv', 'w') as f:
    for c in predicted_candidates:
        pmid = c[0].parent.document.name
        rsid = c[0].get_span()
        phen = c[1].get_span()        
        table_id = c[0].parent.table.position
        row_num = c[0].parent.cell.row.position
        col_num = c[0].parent.cell.col.position # of the rsid
        
        if row_num is None:
            print c[0].parent.cell

        phen = (unravel(pmid, phen, D))
        if isinstance(phen, unicode):
            phen = phen.encode('utf-8')
        
        try:
            log_pval = pval_rsid_dict[pmid][(rsid, table_id, row_num)]
        except KeyError:
            log_pval = -1000
#             continue
        if 10**log_pval > 1e-5: continue

        out_str = '{pmid}\t{rsid}\t{phen}\t{pval}\ttable\t{table_id}\t{row}\t{col}\n'.format(
                    pmid=pmid, rsid=clean_rsid(rsid), phen=phen, pval=log_pval, table_id=table_id, row=row_num, col=col_num)
        f.write(out_str)



