import csv

mesa = [
    {'first': 'joe', 'last': 'lutz', 'address': 'aqui'},
    {'first': 'mike', 'address': 'panera', 'last': 'duncan'},
    {'address': 'nashville', 'first': 'travis', 'last': 'cash'},
]

mesa

columns = ['last', 'first', 'address']

# excel_tab dialect was likely supported on creaky old version
# but required a hyphen instead of underscore.
with open('eggs.csv', 'wb') as csvfile:
    writer = csv.DictWriter(csvfile, columns, dialect='excel-tab')
    for row in mesa:
        writer.writerow(row)

with open('eggs.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, columns, dialect='excel-tab')
    for row in mesa:
        writer.writerow(row)

get_ipython().system('ls -l eggs.csv')

get_ipython().system('cat eggs.csv')

print(repr(open('eggs.csv').read()))

with open('eggs.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, columns, dialect='excel-tab')
    query_results = do_query(FOO_QUERY_SQL)  # This is fakey stub.
    for row in query_results:
        writer.writerow(row)

with open('eggs.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, columns, dialect='excel-tab')
    for row in do_query(FOO_QUERY_SQL):  # This is a fakey stub.
        writer.writerow(row)

import psycopg2

