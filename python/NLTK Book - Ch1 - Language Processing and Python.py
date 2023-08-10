from nltk.book import *

# Any time we want to find out about these texts, 
# we just have to enter their names at the Python prompt:

print(text1)
print(text2)

text1.concordance("monstrous")

print(text1.similar("monstrous"),'\n')
print(text2.similar("monstrous"))

print(text1.common_contexts(["monstrous", "christian"]),'\n')
print(text2.common_contexts(["monstrous", "heartily"]))

text4

text4.dispersion_plot(["citizens","democracy", "freedom","taxes","values"])

#text3.generate()
#The generate() method is not available in NLTK 3.0 
#but will be reinstated in a subsequent version.

text3

len(text3)

len(sorted(set(text3)))

len(set(text3))/len(text3)

text3.count("smote")

text3.count("the")/len(text3)

text1

fdist1 = FreqDist(text1)
print(fdist1)

fdist1.most_common(10)

fdist1['The']

