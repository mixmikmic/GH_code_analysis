import nltk

alice = nltk.corpus.gutenberg.words("carroll-alice.txt")

alice_norm = [word.lower() for word in alice if word.isalpha()]

alice_tags = nltk.pos_tag(alice_norm,tagset="universal")

alice_cfd = nltk.ConditionalFreqDist(alice_tags)

alice_cfd['over']

alice_cfd['spoke']

alice_cfd['answer']

