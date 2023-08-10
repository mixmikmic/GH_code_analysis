import spacy
from spacy import displacy # for visualization
nlp = spacy.load('en')

how_q_ex = 'How can I pay for my orders?'
assertion_ex = 'I make an order by mistake. I won’t pay.'
specific_qex = 'Can I use paypal for order #123?'
when_q_ex = 'When can I get my cashback?'
spelling_ex = 'My order number is #123. How can I pay?'

examples = [how_q_ex, assertion_ex, specific_qex, when_q_ex, spelling_ex]

doc = nlp(assertion_ex)
print(f'text\t|lemma\t|postag\t|tag\t|tag_explain\t\t\t\t   |dep_parser\t   |dep_explain\t\t|stop_word')
for token in doc:
    dep_explain = spacy.explain(token.dep_)
    if type(dep_explain)==type(None):
        dep_explain = 'None'
    
    tag_explain = spacy.explain(token.tag_)   
    print(f'{token.text:<8}|{token.lemma_:<7}|{token.pos_:<7}|{token.tag_:<7}|{tag_explain:42}|{token.dep_:15}|{dep_explain:<25}|{token.is_stop:<7}')

options = {'compact': True, 
           'bg': '#09a3d5',
#            'bg': '#000',
           'color': 'white', 'font': 'Source Sans Pro'}

displacy.render(doc, jupyter=True, style='dep', options=options)

example_sentence = 'James B. Comey, the former F.B.I. director fired by President Trump, said in an ABC News interview that Mr. Trump was “morally unfit to be president,” portraying him as a danger to the nation.'

nlp = spacy.load('en')
doc = nlp(example_sentence)

for chunk in doc.noun_chunks:
    print(f'{chunk.text:<30},{chunk.root.text:<15},{chunk.root.dep_:<7},{spacy.explain(chunk.root.dep_):25},{chunk.root.head.text:<15}')

