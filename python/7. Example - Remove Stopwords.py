import nltk

alice = nltk.corpus.gutenberg.words("carroll-alice.txt")

alice_fd = nltk.FreqDist(alice)

alice_100 = alice_fd.most_common(100)
alice_common = [word[0] for word in alice_100]
common = set(word.lower() for word in alice_common if word.isalpha())

common

descriptive = list(set(common) - set(nltk.corpus.stopwords.words("english")))

descriptive

