print("Hello","World")

print("Hello","World",sep='...',end='!!')

string1='World'
string2='!'
print('Hello' + string1 + string2)

print("Hello %s" % string1)
print("Actual Number = %d" %18)
print("Float of the number = %f" %18)
print("Octal equivalent of the number = %o" %18)
print("Hexadecimal equivalent of the number = %x" %18)
print("Exponential equivalent of the number = %e" %18)

print("Hello %s %s. This meaning of life is %d" %(string1,string2,42))

print('Print width 10: |%10s|'%'x')
print('Print width 10: |%-10s|'%'x') # left justified
print("The number pi = %.2f to 2 decimal places"%3.1415)
print("More space pi = %10.2f"%3.1415)
print("Pad pi with 0 = %010.2f"%3.1415) # pad with zeros

print("Hello World! "*5)

s="hello wOrld"
print(s.capitalize())
print(s.upper())
print(s.lower())
print('|%s|' % "Hello World".center(30)) # center in 30 characters
print('|%s|'% "     lots of space             ".strip()) # remove leading and trailing whitespace
print("Hello World".replace("World","Class"))

s="Hello World"
print("The length of '%s' is"%s,len(s),"characters") # len() gives length
s.startswith("Hello") and s.endswith("World") # check start/end
# count strings
print("There are %d 'l's but only %d World in %s" % (s.count('l'),s.count('World'),s))
print('"el" is at index',s.find('el'),"in",s) #index from 0 or -1

'abc' < 'bbc' <= 'bbc'

"ABC" in "This is the ABC of Python"

s = '123456789'
print('First charcter of',s,'is',s[-3:-1:-1])
print('Last charcter of',s,'is',s[len(s)-1])

print('First charcter of',s,'is',s[-len(s)])
print('Last charcter of',s,'is',s[-1])

print("First three charcters",s[0:3])
print("Next three characters",s[3:6])

print("First three characters", s[:3])
print("Last three characters", s[-3:])

s='012345'
sX=s[:2]+'X'+s[3:] # this creates a new string with 2 replaced by X
print("creating new string",sX,"OK")
sX=s.replace('2','X') # the same thing
print(sX,"still OK")
s[2] = 'X' # an error!!!



