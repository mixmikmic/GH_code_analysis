# Loads data from file into two variables: students and books.
# Each student/book is treated as a dictionary keyed by id/isbn.
# students['<id #>'] yields a dictionary of student first and last name
# books['<isbn #>'] yields a dictionary of title and author

import json

with open('data/test.json') as f:
    data = json.load(f)

books = data['books']
students = data['students']

#Examples
print('Students:\n')
for key, value in students.items():
    print('%s: %s' % (key, value))

print('\nBooks:\n')
for key, value in books.items():
    print('%s: %s' % (key, value))

print('\n')
print('Lookup ISBN 9780394800011:')
print(books['9780394800011'])



