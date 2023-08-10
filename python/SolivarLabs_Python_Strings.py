first_string = 'python_strinG'
second_string = '123'

print(len(first_string))

print(first_string.capitalize())

print(first_string.casefold())

print(first_string.count('n'))

print(first_string.find('t'))

print(first_string.isalnum())
print(second_string.isalnum())


print(first_string.isalpha())
print(second_string.isalpha())

'''Similarly we have, isdecimal , isdigit, isidentifier, islower, isnumeric, isprintable, isspace, istitle, isupper
its good to know about all of them. '''


print(first_string.upper())
print(first_string.lower())

print(first_string.replace('string','strings'))
print(first_string.split())
print(first_string.split('_'))

print('  spaces  '.strip())
print('$$dollars$$'.strip('$'))
#there rstrip and lstrip for right and left only strip

print('Python'.swapcase())
print('all words are important'.title())

print("format has the values to be printed {0} {2} {1}".format("first","second","third"))

print("Hello %(name)s, Good %(wishes)s." % {'name':'John', 'wishes':'Morning'})
print("Hello %(name)s, Good %(wishes)s." % {'name':'Ram', 'wishes':'Night'})

