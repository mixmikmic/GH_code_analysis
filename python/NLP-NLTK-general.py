example_sents = example_text.split('.')
example_sents_toks = [sent.split(' ') for sent in example_sents]

for sent_toks in example_sents_toks:
    print(sent_toks)

import re

example_sents = re.split('(?<=[a-z])[.?!]\s', example_text)
example_sents_toks = [re.split('\s+', sent) for sent in example_sents]

for sent_toks in example_sents_toks:
    print(sent_toks)

import nltk

example_sents = nltk.sent_tokenize(example_text)
example_sents_toks = [nltk.word_tokenize(sent) for sent in example_sents]

for sent_toks in example_sents_toks:
    print(sent_toks)

patterns = [
    (r'.*ing$', 'VBG'),               # gerunds
    (r'.*ed$', 'VBD'),                # simple past
    (r'.*es$', 'VBZ'),                # 3rd singular present
    (r'.*ould$', 'MD'),               # modals
    (r'.*\'s$', 'NN$'),               # possessive nouns
    (r'.*s$', 'NNS'),                 # plural nouns
    (r'.*ly', 'RB'),                  # adverbs
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
    (r'.*', 'NN')                     # nouns (default)
    ]

from nltk import RegexpTagger

regexp_tagger = RegexpTagger(patterns)

sent = nltk.word_tokenize("They refuse to permit us to obtain the refuse permit")
sent_tagged = regexp_tagger.tag(sent)

print(sent_tagged)

brown_tagged_sents = brown.tagged_sents(categories='news')
regexp_tagger.evaluate(brown_tagged_sents)

sent = nltk.word_tokenize("They refuse to permit us to obtain the refuse permit")
sent_tagged = nltk.pos_tag(sent)
print(sent_tagged)

grammar = "NP: {<DT>?<JJ>*<NN.*>+}"

from nltk import RegexpParser

cp = RegexpParser(grammar)

sent_tagged = nltk.pos_tag(sent)
sent_chunked = cp.parse(sent_tagged)

print(sent_chunked)

type(sent_chunked)

sent_chunked.draw()

for node in sent_chunked:
    if type(node)==nltk.tree.Tree and node.label()=='NP':
        phrase = [tok for (tok, tag) in node.leaves()]
        print(' '.join(phrase))

sent_nes = nltk.ne_chunk(sent_tagged)
print(sent_nes)

entities = {'ORGANIZATION':[], 'PERSON':[], 'LOCATION':[]}
for node in sent_nes:
    if type(node)==nltk.tree.Tree:
        phrase = [tok for (tok, tag) in node.leaves()]
        if node.label() in entities.keys():
            entities[node.label()].append(' '.join(phrase))

for key, value in entities.items():
    print(key, value)

grammar = r"""
    NP: {<DT>?<JJ>*<NN.*>+}      # Chunk sequences of DT, JJ, NN
    PP: {<IN><NP>}               # Chunk prepositions followed by NP
    VP: {<VB.*><NP|PP|CLAUSE>+} # Chunk verbs and their arguments
    CLAUSE: {<NP><VP>}           # Chunk NP, VP into a clause
    """
cp = RegexpParser(grammar)

sent_tagged = nltk.pos_tag(sent)
sent_chunked = cp.parse(sent_tagged)
print(sent_chunked)

sent_chunked.draw()

def extract_nps_recurs(tree):
    nps = []
    if not type(tree)==nltk.tree.Tree:
        return nps
    if tree.label()=='NP':
        nps.append(' '.join([tok for (tok, tag) in tree.leaves()]))
    for subtree in tree:
        nps.extend(extract_nps_recurs(subtree))
    return nps

extract_nps_recurs(sent_chunked)

import os
from nltk.parse.stanford import StanfordDependencyParser

os.environ['STANFORD_PARSER'] = '/PATH/stanford-parser-full-2016-10-31'
os.environ['STANFORD_MODELS'] = '/PATH/stanford-parser-full-2016-10-31'

dependency_parser = StanfordDependencyParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
sents_parsed = dependency_parser.parse_sents(sents)

sents_parseobjs = [obj for sent in sents_parsed for obj in sent]

len(sents_parseobjs)

sents_parseobjs[0].tree()

for triple in sents_parseobjs[0].triples():
    print(triple)

print(sents_parseobjs[0].to_conll(10))

print(dir(sents_parseobjs[0]))



