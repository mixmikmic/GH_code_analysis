def unigram_feature(x, unigrams):
    word_list = x.lower().split(" ")
    count = 0
    for unigram in unigrams:
        count += word_list.count(unigram)
    return count
def numeric_feature(x):
    count = 0
    for c in x:
        if x.isnumeric():
            count += 1
    return count
def similarity_feature(x, word):
    word_list = x.lower().split(" ")
    similarity = 0
    for w in word_list:
        for s in wn.synsets(w, pos=wn.NOUN):
            similarity = max(similarity, word.wup_similarity(s)) 
    return similarity
def pos_feature(x, pos):
    word_list = x.lower().split(" ")
    t = nltk.pos_tag(word_list)
    count = 0
    for w in t:
        if w[1] == pos:
                count+=1
    return count
def median_length_feature(x):
    word_list = x.lower().split(" ")
    word_lengths = [len(w) for w in word_list]
    word_lengths.sort()
    return word_lengths[len(word_lengths)//2]
allNames = [name.lower() for name in names.words()]
def names_feature(x):
    word_list = x.lower().split(" ")
    count = 0
    for word in word_list:
        if word in allNames:
            count = 1
    return count

