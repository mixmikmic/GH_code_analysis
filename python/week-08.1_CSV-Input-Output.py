## Creating a list of lists from scratch
# Each 'row' contains a number, its square, and its cube.

exponent_table = []

for i in range(10):
    row = [i, i**2, i**3]
    exponent_table.append(row)

exponent_table

## Writing the list of lists to a CSV file

import csv

out_path = "Exponent_table.csv"

header = ['Number', 'Squared', 'Cubed']

with open(out_path, 'w') as fo:
    csv_writer = csv.writer(fo)
    csv_writer.writerow(header)
    csv_writer.writerows(exponent_table)

## Viewing the contents of the CSV file we just created

get_ipython().system('cat Exponent_table.csv')



## Downloading a CSV containing The Guardian's top 10 'Greatest Arthouse and Drama' films
## https://www.theguardian.com/news/datablog/2010/oct/16/greatest-films-of-all-time

get_ipython().system('wget https://raw.githubusercontent.com/pcda17/pcda17.github.io/master/week/7/Greatest_Arthouse_Drama_Films.csv')

## Loading the CSV as a list of lists

import csv

csv_path = "Greatest_Arthouse_Drama_Films.csv"
list_of_lists = []

with open(csv_path) as fi:
    csv_input = csv.reader(fi)
    for row in csv_input:
        list_of_lists.append(row)

list_of_lists

len(list_of_lists)

## Viewing the header row

list_of_lists[0]

## Viewing a single record's row

list_of_lists[8]

## Viewing the 'director' field in the row we viewed above

row = list_of_lists[8]

row[2]

## Viewing the header row alongside each field's index

list(enumerate(list_of_lists[0]))

## Creating a reduced version of a single row, containing 4 selected fields

row = list_of_lists[9]

film = row[1]
director = row[2]
year = row[4]
country = row[8]

reduced_row = [film, director, year, country]

reduced_row

## Creating a new list of lists containing reduced versions of each row

reduced_list_of_lists = []

for row in list_of_lists:
    film = row[1]
    director = row[2]
    year = row[4]
    country = row[8]
    reduced_row = [film, director, year, country]
    reduced_list_of_lists.append(reduced_row)

reduced_list_of_lists

## Writing the reduced list of lists to a CSV

out_path = "Greatest_Arthouse_Drama_Films_reduced.csv"

with open(out_path, 'w') as fo:
    csv_writer = csv.writer(fo)
    csv_writer.writerows(reduced_list_of_lists)

## Viewing the contents of the CSV file we just created

get_ipython().system('cat Greatest_Arthouse_Drama_Films_reduced.csv')



