#Creating a Dictionary and update/insert

d = {}         #curly braces for dictionary
d[23] = 34 
d["strd"] = 353

print(d)

d = {23:45, "sfj":4924, 1 : 4}
print(d1)


#Fast iteration on Dictionary

for i in d:
    print(i, end = " - ")
    print(d[i])

#Delete element in a dicionary 

del d[23]
d

#Common  Functions
d1 = {}
d1[2] = 1

d2 = {}
d2[2] = 1
 
print(d1 == d2)

print(len(d1))     #returns the length of the dictionary 

d1.clear()
print(d1)      #clears the dictionary 


print(d2.keys())    #returns the list of all keys 
print(d2.values())   #returns the list of all values


23 in d2     #checks if 23 key is in the dictionary or not,  returns boolean

#Leaders in an array 
n = int(input())
l = [int(x) for x in input().strip().split(" ")]

def isLeader(a,l):
    for j in range (a, n):
        if(l[j] > l[a]):
            return False
    return True    

for i in range(n):
    if(isLeader(i,l)):
        print(l[i], end = " ")

        

#Reverse String word wise

l = input().strip().split(" ")
n = len(l)

for i in range(-1, -n-1, -1):
    print (l[i], end = " ")

#Maximise the sum 

n1 = int(input())
l1 = [int(x) for x in input().strip().split(" ")]
n2 = int(input())
l2 = [int(x) for x in input().strip().split(" ")]

maxSum = 0
sum1 = 0
sum2 = 0
i =0
j =0

while i<n1 and j<n2:
    if l1[i] < l2[j]:
        sum1 += l1[i]
        i += 1
    elif l2[j] < l1[i]:
        sum2 +=l2[j]
        j += 1
    else:
        sum1 += l1[i]
        sum2 += l2[j]
        
        if sum1 >sum2 : 
            maxSum += sum1
        else:
            maxSum += sum2
        i += 1
        j += 1
        sum1, sum2 = 0,0

while(i<n1):
    maxSum += l1[i]
    i += 1
while(j<n2):
    maxSum += l2[j]
    j += 1

print(maxSum)

#largest unique substring

s = input()

length = 0
maxStart = 0
maxLength = 0
out = ""

start, end = 0,0
while end < len(s):
    if s[end] in out:
        start = 1 + s.find(s[end])
        out = s[start:end+1]
        end += 1
        
    else:
        out += s[end]
        length += 1
        end +=1
    
    if length > maxLength:
        maxLength = length
        maxStart = start 
    print(out)    
   

print(s[maxStart: maxStart+maxLength])

s = "abcdc"
print(s.find('c'))

