import nltk

f = open('text.txt')
raw = f.read()
tokens = nltk.word_tokenize(raw)
def get_full_text(text):
  every_word = [w for w in text]
  full_text = nltk.Text(every_word)
  return full_text

full_text = get_full_text(tokens)

[w for w in full_text if len(w) > 12]

[w for w in full_text if w.endswith('ente') ]

[w for w in full_text if '–' in w]

len(full_text)

len(set(full_text))

len(set(word.lower() for word in full_text))

len(set(word.lower() for word in full_text if word.isalpha()))

word = 'cat'
if len(word) < 5:
    print('word length is less than 5')

for word in ['Call', 'me', 'Ishmael', '.']:
    print(word)

sent1 = ['Call', 'me', 'Daccio', '.']
for w in sent1:
    if w.endswith('o'):
        print(w)

for token in sent1:
    if token.islower():
        print(token, 'is a lowervcase word')
    elif token.istitle():
        print(token, 'is a titlecase word')
    else:
        print(token, 'is punctuation')

tricky = sorted(w for w in set(full_text) if 'con' in w or 'er' in w)
for w in tricky:
    print(w, end=' ')



