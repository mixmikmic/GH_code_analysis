dictionary = {1:2, 2:3, 4:5}

print({i:dictionary[i] for i in list(dictionary)[:2]})

ind_words = {}
for key, value in iter(dictionary.items()):
    print(key, value)
    ind_words[value]=key
    print(ind_words)

