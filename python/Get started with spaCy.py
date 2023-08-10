import spacy

get_ipython().run_cell_magic('bash', '', 'python -m spacy download en')

text = "This is a sentence. And this is another sentence." 
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
for token in doc:
    print(token, token.pos_) 

text = "This is a sentence. And this is another sentence."
doc = nlp(text)
for sent in doc.sents:
    print([(token, token.pos_) for token in sent]) 



text = "This is a sentence. And this is another sentence." 
doc = nlp(text)
for token in doc:
    print(token, token.pos_, token.dep_, token.head) 



from spacy import displacy
from IPython.core.display import display, HTML
html = displacy.render(doc, style='dep')
display(HTML(html))

text = "Daniel Vila is visiting Austin, Texas"
doc = nlp(text)
for ent in doc.ents:
    print(ent, ent.label_, [token.dep_ for token in ent])

text = "Daniel Vila is visiting Austin, Texas"
doc = nlp(text)
displacy.render(doc, style='ent', jupyter=True)



