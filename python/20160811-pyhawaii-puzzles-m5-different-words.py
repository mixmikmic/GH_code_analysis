def get_words_set(filename):
    return set(word.strip() for word in open(filename))

a = get_words_set('113809of.fic')
b = get_words_set('113809of.rev.2.fic')

len(a), len(b)

a - b

