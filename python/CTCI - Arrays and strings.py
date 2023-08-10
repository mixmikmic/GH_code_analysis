import numpy as np

#Is unique

inputString = input()

print("Method 1:",len(inputString) == len(set(inputString.lower())))

flag = 0
dict = {} #dict = []
for i in inputString:
    i = i.lower()
    if i in dict:
        flag = 1
    else:
        dict[i] = True #dict.append(i)

if flag:
    print("Method 2: Not Unique")
else:
    print("Method 2: Unique")

#check permutation

str1 = input()
str2 = input()

print(sorted(str1.lower()) == sorted(str2.lower()))

#URLify
str1.replace(' ', '%20')

#palindrome permutation
inputString = input()
inputString = inputString.replace(' ', '')

#parse the string and create a character counter map
dict = {}
count = 0
for i in inputString:
    if i in dict:
        dict[i] += 1
    else:
        dict[i] = 1
        
    if dict[i] % 2:
        count += 1
    else:
        count -= 1
        
print(count<=1)

#one away
str1 = input()
str2 = input()

if len(str1) == len(str2):
    #check if a char can be replaced
    count = 0
    for i in range(len(str1)):
        if not str1[i] == str2[i]:
            count += 1
        if count>1:
            print("Can't be replaced")
            break
else:
    if abs(len(str1) - len(str2)):
        #check identical
        if not len(str1) - len(str2):
            temp = str1
            str1 = str2
            str2 = str1
            
        count = 0
        j = 0
        for i in range(len(str1)):
            if not str1[i] == str2[j]:
                count += 1
            else:
                j += 1
            if count == 0 and j == len(str2):
                break
            if count>1:
                print("Can't be inserted/removed")
                break
    else:
        print("Can't be inserted/removed")

#string compression
inputString = input()

if inputString:
    newstring = inputString[0]
    prev = inputString[0]
count = 1
for i in inputString[1:]:
    if i == prev:
        count += 1
    else:
        newstring += str(count)
        newstring += i
        count = 1
        prev = i

newstring += str(count)
print(newstring) if len(newstring) < len(inputString) else print(inputString)

#rotate matrix
numberRows = int(input())
matrix = np.zeros((numberRows, numberRows), dtype=np.int32)

for row in range(numberRows):
    for column in range(numberRows):
        matrix[row, column] = int(input())
        
print(matrix)

#initialize the reversed matrix
reversedMatrix = np.zeros((numberRows, numberRows), dtype=np.int32)

for row in range(numberRows):
    reversedMatrix[:, numberRows - row - 1] = matrix[row, :]
        
print(reversedMatrix)

#rotate layer by Layer if we can't create a new matrix
    

