get_ipython().magic('ls *.txt')

targetText = "Hume Treatise.txt"

with open(targetText, "r") as f:
    theText = f.read()

print("This string has", "{:,}".format(len(theText)), "characters")

import re
theTokens = re.findall(r'\b\w[\w-]*\b', theText.lower())
print(theTokens[:10])

wrds2find = input("What word do you want collocates for?").split(" ") # Ask for the words to search for
print(wrds2find)

contextBefore = 1 # This sets the context of words before the target word
contextAfter = 0 # This sets the context of words after the target word

end = len(theTokens)
counter = 0
theCollocates = []
for word in theTokens:
    if word in wrds2find: # This checks to see if the word is what we want
        for i in range(contextBefore):
            if (counter - (i + 1)) >= 0: # This checks that we aren't at the beginning
                theCollocates.append(theTokens[(counter - (i + 1))]) # This adds words before
        for i in range(contextAfter):
            if (counter + (i + 1)) < end: # This checks that we aren't at the end
                theCollocates.append(theTokens[(counter + (i + 1))]) # This adds words after
    counter = counter + 1
    
print(theCollocates[:30])

print(len(theCollocates))

print(set(theCollocates))

import nltk
tokenDist = nltk.FreqDist(theCollocates)
tokenDist.tabulate(10)

import matplotlib
get_ipython().magic('matplotlib inline')
tokenDist.plot(25, title="Top Frequency Collocates for " + wrd2find.capitalize())

import csv
nameOfResults = wrd2find.capitalize() + ".Collocates.csv"
table = tokenDist.most_common()

with open(nameOfResults, "w") as f:
    writer = csv.writer(f)
    writer.writerows(table)
    
print("Done")



