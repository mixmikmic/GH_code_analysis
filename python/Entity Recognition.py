from spacy.en import English
parser = English()

def show_ents(text):
    
    print("Showing by token ....")
    parsed = parser(text)
    for token in parsed:
        print(token.orth_, token.ent_type_)
    print()
    
    print("Just showing the entities ....")
    entities = list(parsed.ents)
    for entity in entities:
        print(entity.label_,' '.join(t.orth_ for t in entity))
    

# An example
show_ents("I went to New York City and spoke in French.")

# Another example
show_ents("I paid $50 and 50 dollars on March 12, 2016")

# Another example, with an error. German is not listed as a language, but a group
show_ents("I spoke German at Mt. Everest.")

#I O B format
def show_ent_IOB(text):
    parsed = parser(text)
    for token in parsed:
        print(token.orth_, token.ent_type_, token.ent_iob_)
    

# Note that 'New' has B for begin and 'Orleans' has 'I' for inside.
show_ent_IOB("I went to New Orleans to speak French.")

