input_file = open('sample.txt', 'r') # IOError occured because we should put the write true direction.

input_file = open('data/sample.txt', 'r') # True direction, with folder inside.
print(input_file)
input_file.close()

output_file = open('data/mynewfile.txt', 'w') # I used data/name because I want to save in data folder

output_file.close()

input_file = open('data/sample.txt', 'r')
empty_str = ''
line = input_file.readline() 
while line != empty_str:
    print(line)
    line = input_file.readline()

input_file.close()

input_file = open('data/sample.txt', 'r')
for line in input_file:
    print(line)

empty_str= ''
input_file = open('data/sample.txt', 'r')
output_file = open('data/newfile.txt', 'w')
line = input_file.readline()

while line != empty_str:
    output_file.write(line)
    line = input_file.readline()
    
output_file.close()

space = ' '
num_spaces = 0
line = input_file.readline()
for k in range(0, len(line)):
    if line[k] == space:
        num_spaces = num_spaces + 1

num_spaces

s = 'Hello World!'

s.isalpha() #

s.isdigit()

"1".isdigit()

s.islower()

s

s.isupper()

"HELLO WORLD".isupper()

s

s.upper()

s.lower()

s # Does not change... You have to assign it to an new variable or overwrite

s = s.lower()

s

s

s.find('d')

s.find('x')

s

s.replace("l", "*")

s

s.strip('!')

s

s.strip('!').split(" ")

s[:-4]

