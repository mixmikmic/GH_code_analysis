import numpy as np
import collections
from scipy import spatial

filename = 'glove.twitter.27B.200d.txt'

with open(filename,'r') as f:
    lines = f.readlines()
    numWords = len(lines)
    numDimensions = len(lines[200].split(' ')[1:])
    print(numWords, numDimensions)

print(lines[200])
print(lines[201])
print(lines[202])

f.close()

readWords = range(100000)

wordList = []
wordVectorMatrix = np.zeros((len(readWords),numDimensions))
wordVectorDictionary = collections.defaultdict(list)

with open(filename,'r') as f:
    index = 0
    for line in f:
        if index in readWords:
            split = line.split()
            word = (split[0])
            wordList.append(word)
            listValues = map(float, split[1:])
            wordVectorMatrix[index] = listValues
            wordVectorDictionary[word] = listValues
            index += 1
        else:
            break

print('The length of wordList is: %d, and the length of wordVectorDictionary is: %d' %(len(wordList),len(wordVectorDictionary))) 
print('The dimensions of wordVectorMatrix are: %s' %(wordVectorMatrix.shape,))

print('The %d-th word in our word list is: %s' %(len(wordList),wordList[-1]))
print('The first 5 dimensions of "%s" are: %s' %(wordList[-1], tuple(wordVectorMatrix[-1,:5])))
print('Does the vector in wordVectorDictionary for "%s" match the vector in wordVectorMatrix: %s' %(wordList[-1],all(wordVectorDictionary[wordList[-1]] == wordVectorMatrix[-1,:])))

def findClosestWords(word, numWords):
    indexOfWord = wordList.index(word)
    wordVector = wordVectorMatrix[indexOfWord]
    similarityDictionary = {}
    for i in readWords:
        if i == indexOfWord:
            continue
        closeness = 1 - spatial.distance.cosine(wordVector, wordVectorMatrix[i,:])
        similarityDictionary[wordList[i]] = closeness
    for w in sorted(similarityDictionary, key=similarityDictionary.get, reverse=True)[:numWords]:
        print(w, similarityDictionary[w]) 

findClosestWords('please', 10)

def subtractVectors(v1, v2):
    return np.subtract(wordVectorDictionary[v1], wordVectorDictionary[v2])

manWoman = subtractVectors('man','woman')
kingQueen = subtractVectors('king','queen')
brotherSister = subtractVectors('brother','sister')
manKing = subtractVectors('man','king')

print(np.linalg.norm(manWoman, 2))
print(np.linalg.norm(kingQueen, 2))
print(np.linalg.norm(brotherSister, 2))
print(np.linalg.norm(manKing, 2))
print("\n")
print(np.linalg.norm(manWoman - kingQueen, 2))
print(np.linalg.norm(manWoman - brotherSister, 2))
print(np.linalg.norm(kingQueen - brotherSister, 2))
print("\n")
print(np.linalg.norm(manWoman - manKing, 2))
print(np.linalg.norm(kingQueen - manKing, 2))
print(np.linalg.norm(brotherSister - manKing, 2))

findClosestWords('man', 10)
print("\n")
findClosestWords('woman', 10)
print("\n")
findClosestWords('king', 10)
print("\n")
findClosestWords('queen', 10)
print("\n")
findClosestWords('brother', 10)
print("\n")
findClosestWords('sister', 10)

