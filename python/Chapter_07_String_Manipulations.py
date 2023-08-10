# Program to count the occurence of a particular in a string
# Variable declaration

st = "MISSISSIPPI"
ch = "S"
count = 0

print "Enter the string :",st
print "Which character to be counted ?",ch

# Loop to check occurrence of aa character

l = len(st)

for i in st:
    if i == ch:
        count = count + 1

print "The character " + ch + " occurs %d times" % count        

# Program to count the number of vowels in a sentence
# Variable declaration

st = "This is a book"
count = 0

print "Enter the sentence :"
print st

# Loop to count the vowels in the string

for i in st:
    if i == "A":
        count = count + 1
    elif i == "E":
        count = count + 1
    elif i == "I":
        count = count + 1
    elif i == "O":
        count = count + 1
    elif i == "U":
        count = count + 1
    elif i == "a":
        count = count + 1
    elif i == "e":
        count = count + 1
    elif i == "i":
        count = count + 1
    elif i == "o":
        count = count + 1
    elif i == "u":
        count = count + 1

print "%d vowels are present in the sentence" % count
        
        

# Program to test whether a given string is a palindrome or not
# Variable declaration

st = "HYDERABAD"
rst = ""
i = 0
j = len(st) - 1

print "Enter the string :",st

# Palindrome or not 

rst = reversed(st)

if list(st) == list(rst):
    print "%s is a palindrome string " % st
else:
    print "%s is not a palindrome string " % st

# Program to conctenate two strings
# Variable declaration

st1 = "NEW "
st2 = "DELHI"

# input of two strings to be concatenated

print "Enter first string :",st1
print "Enter second string :",st2

# concatenation of two strings
st = st1 + st2

print "Resultant string is ",st    

# Program to compare two strings
# Variable declaration

st1 = "ALPHA"
st2 = "BETA"

print "Enter string 1:",st1
print "Enter string 2:",st2

# compare strings

if (cmp(st1,st2)>0):
    print "%s " + st1 + "is alphabetically greater string"
else:
    print st2 + " is alphabetically greater string"

# Program to sort an array of names in alphabetical order
# Variable declaration

n = 4
names = ["DEEPAK","SHERIN","SONIKA","ARUN"]

print "How many names ?",n
print "Enter the 4 names one by one"
for i in names:
    print i

# Loop to arrange names in alphabetical order

for i in range(0,n-1):
    for j in range(i+1,n):
        if cmp(names[i],names[j])>0:
            
            temp = names[i]
            names[i] = names[j]
            names[j] = temp

print "Names in alphabetical order"
for i in names:
    print i

# Program to convert a line from lower case to upper case
# Variable declaretion

import sys

st = ['l','o','g','i','c','a','l',' ','t','h','i','n','k','i','n','g',' ','i','s',' ','a',' ','m','u','s','t',' ','t','o',' ','l','e','a','r','n',' ','p','r','o','g','r','a','m','m','i','n','g']

print "Enter a sentence :"
for i in st:
    print i,
print     
print "The converted upper case string is"
# loop to convert lower case alphabet to upper case text
for i in range(len(st)):
    if st[i] >= 'a' and st[i] <= 'z':
        st[i] = chr(ord(st[i])-32)


    sys.stdout.write(st[i])

# Program to read a text and to omit al occurrences of a particular word
# Variable declartion

st = "TO ACCESS THE NAME OF THE CITY IN THE LIST"
i = 0
omit = "THE"
l = len(st)
j = 0
word = []
newst = ""
onesps = ""
print "Enter a sentence :"
print st
print "Enter word to omit :",omit

# loop to omit the given word

for i in range(l):
    ch = i
    if ch == ' ':
        for j in word:
            j = " " 
            if j == omit:
                newst = j
                newst = onesps
            j = " "
            j = 0
        else:
            j = ch
            j = j + 1
    i = i + 1

print "After omiting the word " + omit
print newst
print "Press any key to continue"
     

# Program to calculate the amount to be paid for the telegram
# Variable declaration

count = 0
st = "Congratulations on your success in Examinations."
l = len(st)

print "Type the sentence for Telegram"
print st

# loop to count number of words

for i in range(l):
    if st[i] == '?':
        count = count + 1

if count <= 10:
    amt = 5
else:
    amt = 5 + (count - 10) * 1.25

print "Amount to be paid for telegram = Rs.%0.2f" % amt    

# Program to count number of lines,words and characters
# Variable declaration

txt = "What is a string? How do you initialize it? Explain with example.$"
st = ""
i = 0
lns = 0
wds = 0
chs = 0

print "Enter the text, type $ at end."
print txt

# loop to count lines,words and characters in text

while txt[i] != '$':
    # switch case statements
    if txt[i] == ' ':
        wds = wds + 1
        chs = chs + 1
    elif txt[i] == '.':
        wds = wds + 1
        lns = lns + 1
        chs = chs + 1
    elif txt[i] == '?':
        lns = lns + 1
    else: # default
        chs = chs + 1

    i = i + 1

print "Number of char (incl. blanks) =",chs
print "Number of words =",wds
print "Number of lines =",lns


        

