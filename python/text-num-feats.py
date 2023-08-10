# Imports and settings

from string import punctuation

from common.setup_notebook import set_css_style
set_css_style()

print('Give me three sentences')

s1 = input("First sentence: ")
s2 = input("Second sentence: ")
s3 = input("Third sentence: ")

# Concatenate sentences, replace punctuation with space and split on space
# Do the same for each single sentence (for later use)
s = s1 + s2 + s3
for sign in punctuation:
    s = s.replace(sign, ' ')
    s1 = s1.replace(sign, ' ')
    s2 = s2.replace(sign, ' ')
    s3 = s3.replace(sign, ' ')
    
# Create the unique words list
unique_words = list(set(s.split()))

print('unique words are: ', unique_words)

s1_bow, s2_bow, s3_bow = [], [], []

for word in unique_words:
    s1_bow.append(s1.count(word))
    s2_bow.append(s2.count(word))
    s3_bow.append(s3.count(word))

print('First sentence in BoW: ', s1_bow)
print('First sentence in BoW: ', s2_bow)
print('First sentence in BoW: ', s3_bow)



