import csv

outfile = open('my_test.csv', 'wb')
output = csv.writer(outfile)

headers = ['FIRSTNAME', 'LASTNAME', 'CITY']
output.writerow(headers)

output.writerow(['Alex', 'Richards', 'Chicago'])
output.writerow(['John', 'Smith', 'New York'])

outfile.close()

with open('my_test.csv', 'rb') as infile:
    input = csv.reader(infile)
    for row in input:
        print(row)

