import spacy

# load the model
nlp = spacy.load('en_core_web_sm')

#assign avariable with the models output.
doc = nlp("My name is Harrison and I do not likely Apple Music.")
print(doc.text)
for token in doc:
    print(token.text, token.pos_, token.dep_)

doc = nlp("Sometimes I cry myself to sleep at night thinking about Donald Trump and Brexit.")
for token in doc:
    print(token.text, token.pos_, token.dep_)

import pandas as pd
doc = nlp("Chimpanzees drink boba-tea in the sunshine.")
df = pd.DataFrame([token.text for token in doc], columns = ["Text"])
df

