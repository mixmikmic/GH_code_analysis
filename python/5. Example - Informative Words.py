import nltk

alice = nltk.corpus.gutenberg.words("carroll-alice.txt")

alice_fd = nltk.FreqDist(alice)

alice_fd_100 = alice_fd.most_common(100)

moby = nltk.corpus.gutenberg.words("melville-moby_dick.txt")
moby_fd = nltk.FreqDist(moby)
moby_fd_100 = moby_fd.most_common(100)

alice_100  = [word[0] for word in alice_fd_100]
moby_100 = [word[0] for word in moby_fd_100]

list(set(alice_100) - set(moby_100))

list(set(moby_100) - set(alice_100))

