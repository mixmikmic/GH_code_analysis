import spacy

text = u'Gerd suchte ca. 5 min. die 3 Freunde bzw. Kollegen. Sie warteten am 1. Mai in Berlin/ West: am Zoo.'

# load german models
nlp = spacy.load('de')

# annotate text
tokens = nlp(text)

# spaCy has problem with '/' and occasional breaks sentences between words.

for i, s in enumerate(tokens.sents):
    print(i+1, '-->', s)

# print POS tags
for i, s in enumerate(tokens.sents):
    print(i+1, '-->', ' '.join([(token.text + '/' + token.pos_) for token in s]))

def pos_filter(tokens, type='NOUN'):
    return [token for token in tokens if token.pos_ == type]

print('Nouns:')
for i, s in enumerate(tokens.sents):
    print(i+1, '-->', ' '.join([(token.text + '/' + token.pos_) for token in pos_filter(s, 'NOUN')]))

print('Verbs:')
for i, s in enumerate(tokens.sents):
    print(i+1, '-->', ' '.join([(token.text + '/' + token.pos_) for token in pos_filter(s, 'VERB')]))

# print NER tags
for i, s in enumerate(tokens.sents):
    print(i+1, '-->', ' '.join([(token.text + '/' + token.ent_type_) for token in s]))

# extract entities
for entity in tokens.ents:
    print(entity.text, entity.label_)



