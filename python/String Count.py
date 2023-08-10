sentence = "The cat sat on the mat with the rat , cat and rat were playing on the mat and both cat and rat looks very happy"

print(sentence)

print("Sentence : ",sentence)

print("Length of Sentence : ",len(sentence))

type(sentence) #print("Type of Sentence : ",type(sentence))

print("Type of Sentence : ",type(sentence))  #main output looks like <class 'str'>

print("Count No of 'c' :",sentence.count('c'))

print("Count No of 'cat' :",sentence.count('cat'))

print("Count No of :",sentence.count())

print("Count No of ' ' :",sentence.count(' '))

print("Count No of ' ' :",sentence.count(''))

print(len(sentence))

#Split sentence, generates List (collection of elements) 
words = sentence.split()  #without parameter splits using 'space' character
print("Split : ",words)

words = sentence.split('on')
print("Split again : ",words)

print("Type of Words : ",type(words))

words = sentence.split()
print("Words : ",words)
print("")
print("No of Elements in Words : ",len(words))

#Count Number of Elements in Words
bucket={} #declaring empty Dictionary to be used as {'cat':3,...}

"""
loop through each element (say:'word') in List 'words' and append value 1 as default count,
if 'word' is found again and is already inside Bucket, update the count value of that element by 1
"""
for word in words:
    if word in bucket:
        bucket[word] += 1
    else:
        bucket[word] = 1

print("Dictionary : ",bucket)
print('')
print("Type of Bucket : ", type(bucket))

#Sentence
print("Sentence : ",sentence)

print("Occurrences of 'cat': ",bucket['cat'])
print("Occurrences of 'happy': ",bucket['happy'])
print("Occurrences of 'rat': ",bucket['rat'])

from collections import Counter

words = sentence.split()
print("Words : ",words)
print("")
print("No of Elements in Words : ",len(words))

#using Counter
counts = Counter(words)
print("Type counts: ",type(counts))
print('')
print(counts)

print("Occurrences of 'cat': ",counts['cat'])
print("Occurrences of 'happy': ",counts['happy'])
print("Occurrences of 'rat': ",counts['rat'])

print("Using Counter: ")
print(Counter(['a',2, 'b', 'c', 1,'a', 'b', 'd','b','a',1,2,2]))

sentence = "The cat sat on the mat with the rat , cat and rat were playing on the mat and both cat and rat looks very happy"

print(sentence)

countDict = {word:sentence.count(word) for word in sentence.split()}

print(countDict)

