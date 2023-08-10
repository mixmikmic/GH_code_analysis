name = input("Enter your name: ")
print("Hello {name}".format(name=name))

# Lets use list comprehension in the input
some_val = input("Enter something: ")
print("You entered: {}".format(some_val))

try:
    fhandler = open('test.txt', 'w') 
    fhandler.write('Hello World')
except IOError as ex:
    print("Error performing I/O operations on the file: ",ex)
finally:
    if fhandler:
        fhandler.close()

try:
    with open('test.txt', 'w') as fhandler:
        fhandler.write('Hello World')
except IOError as ex:
    print("Error performing I/O operations on the file: ",ex)

try:
    with open('test.txt', 'r+') as fhandler:
        print(fhandler.readline())
        fhandler.writelines(['.', 'This is', ' Python'])
        # Go to the starting of file
        fhandler.seek(0)
        # Print the content of file
        print(fhandler.readlines())
        fhandler.truncate(20)
        fhandler.seek(0)
        print('After truncate: ',fhandler.readlines())
except IOError as ex:
    print("Error performing I/O operations on the file: ",ex)

import csv
row = []
try:
    with open('./sample_datasets/sample_names.csv', 'r') as fh:
        reader = csv.reader(fh)
        for info in reader:
            row.append(info)
except IOError as ex:
    print("Error performing I/O operations on the file: ",ex)

print(row[0:10])

import csv
try:
    with open('test.csv', 'w') as fh:
        writer = csv.writer(fh)
        for num in range(10):
            writer.writerow((num, num**1, num**2))
except IOError as ex:
    print("Error performing I/O operations on the file: ",ex)

import csv
row = []
try:
    with open('./sample_datasets/sample_names.csv', 'r') as fh:
        reader = csv.DictReader(fh)
        for info in reader:
            row.append(info)
except IOError as ex:
    print("Error performing I/O operations on the file: ",ex)

print(row[0:10])

import csv
try:
    fieldnm = ('Title1', 'Title2', 'Title3')
    with open('test_dict.csv', 'w') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnm)
        headers = dict((hdr, hdr) for hdr in fieldnm)
        for num in range(10):
            writer.writerow({'Title1':num, 'Title2':num+1, 'Title3':num+2})
except IOError as ex:
    print("Error performing I/O operations on the file: ",ex)

