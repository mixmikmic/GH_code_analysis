print('Hello World!')

credits()

round(3.14159, 2)

'Hello World!'.replace('e', 'o')

3 + 3

'3' + '3'

int('3') + int('3')

'Hello ' + 'World' + '!'

name = 'Frederik'

'Hello ' + name + '!'

name = 'Sarah'

'Hello ' + name + '!'

def f(x):
    return x**2

f(3)

f(4)

def hello(name):
    return 'Hello ' + name + '!'

hello('hackers')

def new_mail(name, number):
    return ''

new_mail('Frederik', 3)  # => 'Hello Frederik, you have 3 new e-mails.'

names = ['Maja', 'Willi', 'Flip']

names[0]

len(names)

length = len(names)
position = length - 1  # because, you know, 0 and stuff.
names[position]

names[len(names) - 1]

names[-1]

animals = {'Maja': 'Bee',
           'Willi': 'Bee',
           'Flip': 'Grasshopper'}

animals['Maja']

def hello(name):
    return 'Hello ' + name + ', you are a ' + animals[name] + '!'

hello('Flip')

for name in names:
    print('Hello ' + name + '!')

for i, name in enumerate(names, start=1):  # Otherwise, it would start with 0 again, which is less pretty.
    print(str(i) + '. ' + name)

if len(names) == 2:
    print('I see, there are two of you.')
if len(names) == 3:
    print('I see, there are three of you.')
if len(names) == 4:
    print('I see, there are four of you.')

if len(names) < 3:
    print('You are too few!')
elif len(names) == 3:
    print('You are exactly the right number!')
else:
    print('You are too many!')

for name in names:
    if len(name) > 4:
        print(name)
    else:
        print('[Too short …]')

# Enter your code here.

