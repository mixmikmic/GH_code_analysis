languages = {
    'scripting': ['Shell', 'Python', 'Perl'],
    'object oriented': ['C++', 'Java', 'Python'],
    'compiled': ['C', 'C++', 'Java']
}

languages.keys()

languages.keys()[0]

for category in languages.keys():
    print(category)

for item in languages.values():
    print(item)

for cat, lang in languages.items():
    print(cat, '...', lang)

firstname = 'Hello'
lastname = 'world'

firstname, lastname = 'Hello', 'world'
firstname

a, b, c = 0, 1

a, b = 1, 4, 7

(4, 6, 7)

m, n = (88, 99)
m

for item in languages.items():
    print(item)
    cat, lang = item #'object oriented', ['C++', 'Java', 'Python']
    print(cat, '....')

for cat, lang in languages.items():
    print(cat, '....')

languages.items()

for item in [1, 2, 3, 4]:
    print("The value is {}".format(item))

for item in [('a', 1), ('b', 2), ('c', 3), ('d', 4)]:
    print("The value is {}".format(item))

for item in [('a', 1), ('b', 2), ('c', 3), ('d', 4)]:
    print("The key is {} value is {}".format(item[0], 
                                             item[1]))

for item in [('a', 1), ('b', 2), ('c', 3), ('d', 4)]:
    key, val = item[0], item[1]
    print("The key is {} value is {}".format(key, 
                                             val))

for item in [('a', 1), ('b', 2), ('c', 3), ('d', 4)]:
    key, val = (item[0], item[1])
    print("The key is {} value is {}".format(key, 
                                              val))

for item in [('a', 1), ('b', 2), ('c', 3), ('d', 4)]:
#     print((item[0], item[1]))
#     print(item)
    key, val = item
    print("The key is {} value is {}".format(key, 
                                              val))

for key, val in [('a', 1), ('b', 2), ('c', 3), ('d', 4)]:
    print("The key is {} value is {}".format(key, 
                                              val))

for cat, langs in languages.items():
    if 'Python' in langs:
        print('Python is in {}'.format(cat))
    else:
        print('Python is not in {}'.format(cat))

for cat, langs in languages.items():
    if 'Python' in langs and 'Perl' in langs:
        print('Python and Perl are in {}'.format(cat))
    elif 'Python' in langs:
        print('Python is in {}'.format(cat))
    else:
        print('Python is not in {}'.format(cat))

0 or 1

1 or 2

bool(0)

a = 0 or 1

b = 0 if 0 else 5

b

b = None
if 0:
    b = 0
else:
    b = 5
b

bool([])

lst = [1, 2, 3]
if lst:
    print(lst)

'' == False

bool('') == False

'a' in 'apple'

'a' is 'a'

'a' is str

'a' is 'b'

'a' is str('a')

2 < 3

1 < 2 < 3









