import nltk

stories = nltk.corpus.gutenberg.words("bryant-stories.txt")
tags = nltk.pos_tag(stories, tagset="universal")

tags[:10]

for ((word1,tag1),(word2,tag2),(word3,tag3)) in nltk.trigrams(tags):
    if tag1 ==  "NOUN" and word2 == "or" and tag3 == "NOUN":
        print word1 + " " + word2 + " " + word3

