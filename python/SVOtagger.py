# keep postags as two list of strings : words and postags
# create 3 sets of postag for entire dataset 
# check difference in number of words between 3 sets
# and do svo-split . . .then check difference in verb block, which should determine the split


import spacy
spacy_nlp = spacy.load('en')               # larger alternative 'en_core_web_lg'
spacytkn = lambda sentstr : spacy_nlp(sentstr)
# takes a string sentence and returns list of tokens which are 
# class with postag in token.pos_, token.text etc

import nltk

from pycorenlp import StanfordCoreNLP
scnlp_nlp = StanfordCoreNLP('http://localhost:9000')
scnlptkn  = lambda sentstr : scnlp_nlp.annotate(sentstr, properties = {
                                        'annotators': 'tokenize,ssplit,pos,depparse,parse',
                                        'outputFormat': 'json' })['sentences'][0]
scnlp_tree   = lambda sentstr : scnlptkn(sentstr)['parse']+'\n'
scnlp_tree_strlist   = lambda strlist : [scnlp_tree(sentstr)   for sentstr in strlist]

# >> note : 
# >> scnlp_nlp.annotate takes a string input and gives dictionary output
# >> scnlp_tree extracts dictionary value (which is a string) for key = 'parse' and add '\n'

import pprint
prettyprinter = pprint.PrettyPrinter(indent=1)
oprint = lambda inobj : prettyprinter.pprint(inobj)

bprint = lambda s : print(bytes(s,encoding='utf-8'))
ex_utf=[i for i in range(0,32,1)] + [i for i in range(127,255,1)] # define non utf characters
notutf8 = lambda s : any(c in ex_utf for c in bytes(s, encoding='utf-8'))  
flatten = lambda list_of_list : [val for sublist in list_of_list for val in sublist]

import numpy as np

fname_sts_trn = '../Proj_Me1/data/sts_source/sts-train.txt'
fname_sts_val = '../Proj_Me1/data/sts_source/sts-dev.txt'
fname_sts_tes = '../Proj_Me1/data/sts_source/sts-test.txt'

def read_sts(fname):

    f = open(fname, 'r', encoding='utf-8')

    i = 0
    scores = []
    sentences1 = []
    sentences2 = []
    while True:
        l = f.readline()  # read 1 line
        if not l:
            break

        data_fields = l.split("\t")
        scores.append(float(data_fields[4]))
        sentences1.append(data_fields[5])
        sentences2.append(data_fields[6])
        i = i + 1
    
        
    f.close()
    print(fname,"records count:",i)

    return i, scores, sentences1, sentences2

# read data files and obtain lists of sentences
n_trn, scores_trn, sentences1_trn, sentences2_trn = read_sts(fname_sts_trn)
n_val, scores_val, sentences1_val, sentences2_val = read_sts(fname_sts_val)
n_tes, scores_tes, sentences1_tes, sentences2_tes = read_sts(fname_sts_tes)

n_data = n_trn + n_val + n_tes
sent1_all = sentences1_trn + sentences1_val + sentences1_tes
sent2_all = sentences2_trn + sentences2_val + sentences2_tes
sent_all = sent1_all + sent2_all

print("Total data samples:",n_data)

## correction in data to prevent nltk from crashing
sent_all[2918] = sent_all[2918].replace('\x12',"'")
bprint(sent_all[2918])

# To read and return list of strings, each string is the tree for a sentence
# Each tree has been written in multiple lines in the file
# Beginning of each tree is the string firstline

firstline = '(ROOT\n'
def read_trees(fname):

    f = open(fname, 'r', encoding='utf-8')
    flines = f.readlines()
    f.close()
    print(len(flines))

    tree = flines[0]  # each tree is a single string with multiple \n characters
    treelist = []     # to store list of tree
    assert (tree == firstline)  # check that first line is the string '(ROOT\n'

    for line in flines[1:]:         # read from second line onwards
        if line == firstline:       # signifies new tree detected
            treelist.append(tree)   # append last tree to treelist
            tree = line             # reset tree to first line
        else:
            tree = tree + line

    treelist.append(tree)           # append last tree after exiting loop

    return treelist

sent_trees = read_trees('./data/scnlp_sent_trees.txt')
assert(len(sent_trees) == (n_data*2))

def get_tup(tree, brac_start, brac_end):  # splits substring in tree[brac_start+1:brac_end]
                                          # into tupple of 2 strings 
                                          # e.g. '(VBZ run' into ('VBZ','run')
    if brac_end >= brac_start + 3:
        sstr = tree[brac_start+1:brac_end]
        try:
            stup = sstr.split(' ')
            tup = (stup[1],stup[0])
        except:
            tup = (None,sstr)
    else:
        tup = None     # or (None,None)
    return tup

def get_next_idx(inlist, idx, item):  # search for location where item next occur in input list
    try:
        next_idx = inlist[idx+1:].index(item) + idx + 1
        endoftree = False
    except:
        endoftree = True
        next_idx = idx    # set to end of list length originally
    return endoftree, next_idx
        
def scnlp_postag(tree):  # extract list of tuples of (word, postag) from tree generated scnlp
    treelen = len(tree)
    if treelen < 5:
        return []
    endoftree = False
    tuplist = []
    next_brac_start = -1
    brac_end = -1
    while not endoftree:
        endoftree, brac_end = get_next_idx(tree,next_brac_start,')') 
        while not endoftree and next_brac_start < brac_end:
            brac_start = next_brac_start
            endoftree, next_brac_start = get_next_idx(tree,brac_start,'(')
        if brac_start < brac_end and brac_start >= 0:
            tup = get_tup(tree, brac_start, brac_end)
            #print(tup)
            tuplist.append(tup)
            brac_start = next_brac_start

    return tuplist
    

# splits list of format [('Eyes', 'NNP'), ('are','VBP') . . ]
# into 2 lists : ['Eyes','are' . . ] and ['NNP', 'VBP' . . ]
def split_postaglist(postaglist):
    return [postag[0] for postag in postaglist], [postag[1] for postag in postaglist]

VB_tags = ['VB','VBD','VBG','VBN','VBP','VBZ']
verbloc = np.zeros((n_data*2,3,2),dtype=np.int)
def find_verbloc(pos):  # find location of first block of verbs, pos is list of pos strings
    found = False
    poslen = len(postag)
    if poslen == 0:
        return found, None, None
    vstart = 0
    vend = 0
    while ((vstart+1) < poslen) and (pos[vstart] not in VB_tags):
        vstart = vstart + 1
    if (vstart < poslen) and (pos[vstart] in VB_tags):
        found = True
        vend = vstart
        while ((vend+1) < poslen) and (pos[vend] in VB_tags):
            vend = vend + 1
    return found, vstart, vend

# create array of tokenized words and pos
tmp_pos = [list()] * n_data * 2
sent_pos = [tmp_pos] * 3
sent_w = [tmp_pos] * 3

#for i in range(n_data*2):
    

n = 10645
testsent = sent_all[n]
# spacy postag
doc = spacy_nlp(testsent)
print("SpaCy postag")
spacy_postag = [(token.text,token.tag_) for token in doc]
print(spacy_postag)

# nltk postag
stkn = nltk.word_tokenize(testsent)
spostag = nltk.pos_tag(stkn)
print("NLTK postag")
print(spostag)

# scnlp postag
print("SCNLP postag")
tree = sent_trees[n]
#print(tree)
print(scnlp_postag(tree))
print(spacy_postag==scnlp_postag(tree))



print(sent_all[n])
print(tree)

testsent = 'A man is spreading shredded cheese on a pizza.'
stkn = nltk.word_tokenize(testsent)
spostag = nltk.pos_tag(stkn)
print(spostag)
print(scnlp_parse(testsent))















# Use stanford corenlp parser to obtain sentence tree so as to obtain noun phrase and verb phrase
# Only need to do this once correctly
# sentence tree is written to file and can be read back when needed
# parsing routine takes 10-15 mimutes for 17k sentences

sent_trees = scnlp_tree_strlist(sent_all)

# input trees is a list of strings, each string correspond to one sentence tree
def write_trees(fname, strlist):    
    f = open(fname, 'w', encoding='utf-8')
    for s in strlist:
        f.write(s)
    f.close()

assert(len(sent_trees) == (n_data*2))
write_trees('./data/scnlp_sent_trees.txt',sent_trees)

# Check sentences where second line of tree is not '  (S'
# Checked : first line is always 'ROOT(\n'
# A tree has at most 160 lines
# Note str.split('\n') function deletes the \n ending

secondline = '  (S'             
def styp_truncate(s):
    l = len(s)
    i = 3
    ans = s[0:3]
    try:
        nextbrac_idx = s[3:].index('(')+3
        return s[:nextbrac_idx+1]
    except:
        return s

def get_nonsenttypes(trees):
    count = 0
    senttypes = []
    senttypes_count = []
    senttypes_idx = []
    for i,tree in enumerate(trees):
        lines = tree.split('\n')
        if lines[1] != secondline:
            count = count + 1
            styp = lines[1]
            # if styp is more than 2 open brackets, truncate till next open bracket, else unchange
            styp = styp_truncate(styp)
            if styp not in senttypes:
                senttypes.append(styp)
                senttypes_count.append(1)
                senttypes_idx.append([i])
            else:
                senttypes_count[senttypes.index(styp)] += 1
                senttypes_idx[senttypes.index(styp)].append(i)
    return count, senttypes, senttypes_count, senttypes_idx


def print_sent(styp, senttypes, senttypes_count, senttypes_idx, trees, sents):
    print("Printing sentences with beginning of type : ", styp)
    try:
        i = senttypes.index(styp)
        for j in senttypes_idx[i]:
            print(j,":",sents[j])
            #print(trees[j])
    except:
        print(styp,"is not in the list")


count, senttypes, senttypes_count, senttypes_idx = get_nonsenttypes(sent_trees)
print(sum(senttypes_count))
print("Atypical sentence count     : %d, %.1f%%" %(count,count/n_data/2*100))
print("Types of atypical sentences :",len(senttypes))
for i in range(len(senttypes)):
    print(senttypes_count[i],":", senttypes[i])


styp = '  (X'
print_sent(styp, senttypes, senttypes_count, senttypes_idx, (sent_trees), sent_all)


# extract non-utf sequences

def extract_non_utf8seq(instr):
    bstr = bytearray(instr, encoding='utf-8')

    ex_utf_idx = [i for i,c in enumerate(bstr) if c in ex_utf]
    seq_list=[]
    if len(ex_utf_idx) >0:
        lasti = ex_utf_idx[0]
        seq = bytearray('',encoding='utf-8')
        seq.append(bstr[lasti])
        for i in ex_utf_idx[1:]:
            if i==(lasti+1):
                seq.append(bstr[i])
            else:
                seq_list.append(seq)
                seq = bytearray('',encoding='utf-8')
                seq.append(bstr[lasti])
            lasti = i
        seq_list.append(seq)
    return seq_list

badsent_idx=[]
badsent_nonutf=[]
for i,s in enumerate(sent1_all):
    if notutf8(s) or notutf8(sent2_all[i]):
        badsent_idx.append(i)
        badsent_nonutf.append([extract_non_utf8seq(s) + extract_non_utf8seq(sent2_all[i])])
print("Number of samples with non-utf8 sequences :", len(badsent_idx))

nonutf_seq_npy = np.asarray(flatten(badsent_nonutf))
nonutf_unique = np.unique(nonutf_seq_npy)
print("Number of unique nonutf sequences :", nonutf_unique.shape[0])
for c in nonutf_unique:
    print(c)


# Print sentences with specific non-utf sequences

n = 2
seqlist = nonutf_unique[n]
print("Sentences with sequence",seqlist,":")
for i, nonutf_list in enumerate(badsent_nonutf):
    if seqlist in nonutf_list:
        print("----------------")
        print("Sample ",i)
        bprint(sent1_all[badsent_idx[i]])
        bprint(sent2_all[badsent_idx[i]])

print("----------------")
m=2505
snew = sent1_all[m].replace('\xa3',' ')
print(len(snew),len(sent1_all[m]))
bprint(snew)

