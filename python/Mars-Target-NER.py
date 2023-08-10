from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
from pycorenlp import StanfordCoreNLP
from codecs import open as copen
from collections import defaultdict as ddict
from csv import DictWriter
import sys
from copy import copy
import time
from pprint import pprint
import re

import os, glob
import pickle

print(sklearn.__version__)

#accept_labels = set(['Element', 'Mineral', 'Target', 'Material', 'Locality', 'Site'])
accept_labels = set(['Target', 'Mineral', 'Element'])

class BratToCRFSuitFeaturizer(object):
    def __init__(self, corenlp_url='http://localhost:9000', iob=False):
        '''
        Create Converter for converting brat annotations to Core NLP NER CRF
        classifier training data.
        @param corenlp_url: URL to corenlp server.
                To start the server checkout: http://stanfordnlp.github.io/CoreNLP/corenlp-server.html#getting-started
        @param iob: set 'True' for IOB encoding
        '''
        self.corenlp = StanfordCoreNLP(corenlp_url)
        self.iob = iob

    def convert(self, text_file, ann_file):
        text, tree = self.parse(text_file, ann_file)
        props = { 'annotators': 'tokenize,ssplit,lemma,pos,ner', 'outputFormat': 'json'}
        if text[0].isspace():
            text = '.' + text[1:]
            # Reason: some tools trim/strip off the white spaces
            # which will mismatch the character offsets
        output = self.corenlp.annotate(text, properties=props)
        records = []
        for sentence in output['sentences']:
            sent_features = []
            continue_ann, continue_ann_en = None, None
            for tok in sentence['tokens']:
                begin, tok_end = tok['characterOffsetBegin'], tok['characterOffsetEnd']
                label = 'O'
                if begin in tree:
                    node = tree[begin]
                    if len(node) > 1:
                        print("WARN: multiple starts at ", begin, node)
                        if tok_end in node:
                            node = {tok_end: node[tok_end]} # picking one
                            print("Chose:", node)

                    ann_end, labels = list(node.items())[0]
                    if not len(labels) == 1:
                        print("WARN: Duplicate labels for token: %s, label:%s.                              Using the first one!" % (tok['word'], str(labels)))
                    if accept_labels is not None and labels[0] in accept_labels:
                        label = labels[0]

                    if tok_end == ann_end: # annotation ends where token ends
                        continue_ann = None
                    elif tok_end < ann_end and label != 'O':
                        #print("Continue for the next %d chars" % (ann_end - tok_end))
                        continue_ann = label
                        continue_ann_end = ann_end 
                    if label != 'O' and self.iob:
                        label = "B-" + label
                elif continue_ann is not None and tok_end <= continue_ann_end:
                    #print("Continuing the annotation %s, %d:%d %d]" % 
                    #(continue_ann, begin, tok_end, continue_ann_end))
                    label = continue_ann            # previous label is this label
                    if continue_ann_end == tok_end: # continuation ends here
                        #print("End")
                        continue_ann = None
                    if self.iob:
                        label = "I-" + label
                sent_features.append([tok['word'], tok['lemma'], tok['pos'], tok['ner'], label])
            yield sent_features

    def parse(self, txt_file, ann_file):
        with copen(ann_file, 'r', encoding='utf-8') as ann_file:
            with copen(txt_file, 'r', encoding='utf-8') as text_file:
                texts = text_file.read()
            anns = map(lambda x: x.strip().split('\t'), ann_file)
            anns = filter(lambda x: len(x) > 2, anns)
            # FIXME: ignoring the annotatiosn which are complex

            anns = filter(lambda x: ';' not in x[1], anns)
            # FIXME: some annotations' spread have been split into many, separated by ; ignoring them

            def __parse_ann(ann):
                spec = ann[1].split()
                name = spec[0]
                markers = list(map(lambda x: int(x), spec[1:]))
                #t = ' '.join([texts[begin:end] for begin,end in zip(markers[::2], markers[1::2])])
                t = texts[markers[0]:markers[1]]
                if not t == ann[2]:
                    print("Error: Annotation mis-match, file=%s, ann=%s" % (txt_file, str(ann)))
                    return None
                return (name, markers, t)
            anns = map(__parse_ann, anns) # format
            anns = filter(lambda x: x, anns) # skip None

            # building a tree index for easy accessing
            tree = {}
            for entity_type, pos, name in anns:
                if entity_type not in accept_labels:
                    continue
                begin, end = pos[0], pos[1]
                if begin not in tree:
                    tree[begin] = {}
                node = tree[begin]
                if end not in node:
                    node[end] = []
                node[end].append(entity_type)

            # Re-read file in without decoding it
            text_file = copen(txt_file, 'r', encoding='utf-8')
            texts = text_file.read()
            text_file.close()
            return texts, tree

def scan_dir(dir_name):
    items = glob.glob(dir_name + "/*.ann")
    items = map(lambda f: (f, f.replace(".ann", ".txt")), items)
    return items

def preprocess_all(list_file, out_file):
    featzr = BratToCRFSuitFeaturizer(iob=True)
    tokenized = []
    with open(list_file) as f:
        examples = map(lambda l:l.strip().split(','), f.readlines())
    for txt_file, ann_file in examples:
        sents = featzr.convert(txt_file, ann_file)
        tokenized.append(list(sents))

    pickle.dump(tokenized, open(out_file, 'wb'))
    print("Dumped %d docs to %s" % (len(tokenized), out_file))

#######################
# Evaluates the model
def evaluate(tagger, corpus_file):
    
    corpus = pickle.load(open(corpus_file, 'rb'))
    y_pred = []
    y_true = []
    for doc in corpus:
        seq = merge_sequences(doc)
        truth = seq2labels(seq)
        preds = tagger.tag(seq2features(seq))
        assert len(truth) == len(preds)
        y_true.extend(truth)
        y_pred.extend(preds)    
    assert len(y_true) == len(y_pred)
    table = ddict(lambda: ddict(int)) 
    for truth, pred in zip(y_true, y_pred):
        table[truth][pred] += 1
        table[truth]['total'] += 1
        table['total'][pred] += 1
        table['total']['total'] += 1
    keys = []
    for label in accept_labels:
        keys.append('B-%s' % label)
        keys.append('I-%s' % label)
    col_keys = copy(keys)
    precision, recall = {}, {}
    for k in set(keys):
        tot_preds = table['total'][k]
        tot_truth = table[k]['total']
        table['Precision'][k] = "%.4f" % (float(table[k][k]) / tot_preds) if tot_preds else 0 
        table['Recall'][k] = "%.4f" % (float(table[k][k]) / tot_truth) if tot_truth else 0 
    col_keys.extend(['O', 'total'])
    keys.extend(['', 'Precision', 'Recall', '', 'O', 'total'])
    return table, keys, col_keys


def printtable(table, row_keys, col_keys, delim=','):
    """
    print table in CSV format which is meant to be copy pasted to Excel sheet 
    """
    f = sys.stdout
    out = DictWriter(f, delimiter=delim, restval=0, fieldnames=col_keys)
    f.write("%s%s" % ("***", delim))
    out.writeheader()
    for k in row_keys:
        if not k.strip():
            f.write("\n")
            continue
        f.write("%s%s" % (k, delim))
        out.writerow(table[k])
    f.write("\n")

p_dir = "/Users/thammegr/work/mte/data/newcorpus/workspace"
train_list = p_dir + "/train_62r15_685k14_384k15.list"
dev_list= p_dir + "/development.list"
test_list = p_dir + "/test.list"

train_corpus_file = 'mte-corpus-train.pickle'
preprocess_all(train_list, train_corpus_file)

# Test and Development set
dev_corpus_file = 'mte-corpus-dev.pickle'
preprocess_all(dev_list, dev_corpus_file)
test_corpus_file = 'mte-corpus-test.pickle'
preprocess_all(test_list, test_corpus_file)

corpus_file = 'mte-corpus-train.pickle'
corpus = pickle.load(open(corpus_file, 'rb'))
corpus[0][10]

#%%time
config = {
    'POS': False,
    'gen_POS': True, # generalize POS
    'bias': True,
    'max_suffix_chars': 3,
    'is_lower': True,
    'is_upper': True,
    'is_title': True,
    'text': True,
    'wordshape': 'sound',
    'NER': False, # default NER
    'context': list(range(-1, 2))
}

def get_wordshape_general(word):
    """
    Makes shape of the word based on upper case, lowercase or digit
    """
    # Note : the order of replacement matters, digits should be at the last
    return re.sub("[0-9]", 'd', 
                  re.sub("[A-Z]", 'X',
                         re.sub("[a-z]", 'x', word)))

def get_wordshape_sound(word):
    """
    Makes shape of word based on the vowel or consonenet sound
    """
    # Note : the order of replacement matters, c, v, d in order
    word = re.sub("[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]", 'c', word) # consonents
    word = re.sub("[AEIOUaeiou]", 'v', word) # vowels
    word = re.sub("[0-9]", 'd', word) # digits
    return word

def get_wordshape_sound_case(word):
    """
    Makes shape of word based on the vowel or consonenet sound considering case
    """
    word = re.sub("[bcdfghjklmnpqrstvwxyz]", 'c', word) # consonents
    word = re.sub("[BCDFGHJKLMNPQRSTVWXYZ]", 'C', word) # upper consonents
    word = re.sub("[aeiou]", 'v', word) # vowels
    word = re.sub("[AEIOU]", 'V', word) # upper vowels
    word = re.sub("[+-]?[0-9]+(\.[0-9]+)?", 'N', word) # digits
    return word

def word2features(sent, idx):
    word = sent[idx]
    words = []
    feats = []

    # Context
    context = set(config.get('context', []))
    context.add(0)  # current word
    for ctx in sorted(context):
        pos = ctx + idx
        if pos >= 0 and pos < len(sent):
            words.append((str(ctx), sent[pos]))
    
    if idx == 0:
        feats.append('BOS') # begin of sequence
    if idx == len(sent) - 1:
        feats.append('EOS')
    for prefix, word in words:
        assert len(word) == 5
        txt, lemma, POS, ner, label = word 
        if config.get('bias'):
            feats.append('%sword.bias'% (prefix))
        if config.get('POS'):
            feats.append('%sword.pos=%s' %(prefix, POS))
        if config.get('gen_POS'):
            feats.append('%sword.genpos=%s' %(prefix, POS[:2]))
        if config.get('max_suffix_chars'):
            for i in range(1, config.get('max_suffix_chars', -1) + 1):
                if len(txt) < i:
                    break
                feats.append('%sword[-%d:]=%s' % (prefix, i, txt[-i:]))
        if config.get('is_lower'):
            feats.append('%sword.islower=%s' % (prefix, txt.islower()))
        if config.get('is_upper'):
            feats.append('%sword.isupper=%s' % (prefix, txt.isupper()))
        if config.get('is_title'):
            feats.append('%sword.istitle=%s' % (prefix, txt.istitle()))
        if config.get('wordshape'):
            shape = config['wordshape'] 
            if shape == 'general':
                shape_val = get_wordshape_general(txt)
            elif shape == 'sound':
                shape_val = get_wordshape_sound(txt)
            elif shape == 'sound_case':
                shape_val = get_wordshape_sound_case(txt)
            else:
                raise Error("Word Shape spec unknown '%s'" % config['wordshape'])
            feats.append('%sword.shape=%s' % (prefix, shape_val))
        if config.get('NER'):
            feats.append('%sword.ner=%s' % (prefix, ner))
        if config.get('text'):
            feats.append('%sword.text=%s' % (prefix, txt))
    return feats

def seq2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def seq2labels(sent):
    # Assumption the last one in array is always a label
    return [tok[-1] for tok in sent] 

def merge_sequences(doc):
    '''
    document contains multiple sentences. here all sentences in document are merged to form one large sequence.
    '''
    res = []
    for seq in doc:
        res.extend(seq)
        res.append(['|', '|', '|', 'O', 'O']) # sentence end marker
    return res
  

def train(corpus, model_file):
    trainer = pycrfsuite.Trainer(verbose=False)
    # Load training examples
    flag = True
    for doc in corpus:
        seq = merge_sequences(doc)
        x_seq = seq2features(seq)
        if flag:
            p = 403
            print("Sample features:")
            print("\n".join(map(str, seq[p-6:p+6])))
            print("\n".join(x_seq[p]))
            flag = False
        y_seq = seq2labels(seq)
        trainer.append(x_seq, y_seq)

    trainer.set_params({
        'c1': 0.5,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })


    st = time.time()
    pprint(trainer.params())
    pprint(config)
    trainer.train(model_file)
    print("Training Time: %.3fs" % (time.time() - st))

model_file = 'jpl-mars-target-ner-model.crfsuite'
train(corpus, model_file)

tagger = pycrfsuite.Tagger()
tagger.open(model_file)
print("\nEvaluating on Development Set\n")
dev_corpus_file = 'mte-corpus-dev.pickle'
printtable(*evaluate(tagger, dev_corpus_file))

print("\nEvaluating on Test Set\n")
test_corpus_file = 'mte-corpus-test.pickle'
printtable(*evaluate(tagger, test_corpus_file))

tagger = pycrfsuite.Tagger()
tagger.open(model_file)

with open(dev_corpus_file, 'rb') as f:
    dev_corpus = pickle.load(f)

ctx = (-3, 4)
c = 0
print("idx, Truth, Predicted, Word, Comment ")
for doc in dev_corpus:
    seq = merge_sequences(doc)
    y = seq2labels(seq)
    y_ = tagger.tag(seq2features(seq))
    
    for idx in range(len(seq)):
        a, p, tok = y[idx], y_[idx], seq[idx]
        if a == 'O' and p == 'B-Element':
            for pos in filter(lambda p: 0 <= p < len(seq), range(idx+ctx[0], idx+ctx[1])):
                if idx == pos:
                    label = "<CORR>" if a == p else "<ERR>"
                else:
                    label = "%d" % (pos - idx)
                print("%4d %9s %9s %8s %s" % (pos, y[pos], y_[pos], label, str(seq[pos])))
            print("")
            if a != p:
                c += 1
print(c)



print("\nTest Set")
test_corpus_file = 'mte-corpus-test.pickle'
printtable(*evaluate(tagger, test_corpus_file))



def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.
    
    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    #tagset.append('O')
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

def evaluate(tagger, corpus_file):    
    corpus = pickle.load(open(corpus_file, 'rb'))
    y_pred = []
    y_true = []
    for doc in corpus:
        seq = merge_sequences(doc)
        y_true.append(seq2labels(seq))
        y_pred.append(tagger.tag(seq2features(seq)))
    return bio_classification_report(y_true, y_pred)


dev_corpus_file = 'mte-corpus-dev.pickle'
test_corpus_file = 'mte-corpus-test.pickle'
print("Development")
print(evaluate(tagger, dev_corpus_file))

print("Testing")
print(evaluate(tagger, test_corpus_file))

from collections import Counter
info = tagger.info()

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

print("Top likely transitions:")
print_transitions(Counter(info.transitions).most_common(15))

print("\nTop unlikely transitions:")
print_transitions(Counter(info.transitions).most_common()[-15:])

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-6s %s" % (weight, label, attr))    

print("Top positive:")
print_state_features(Counter(info.state_features).most_common(20))

print("\nTop negative:")
print_state_features(Counter(info.state_features).most_common()[-20:])



arr = ['a', 'b', 'c']
a, b, c =  *arr
print(a,b,c)


s = "hellow 124.45 -65.7623"
get_wordshape_sound_case("hellow 124.45 -65.7623")
#get_wordshape_sound(s)

"abcd"[:2]



