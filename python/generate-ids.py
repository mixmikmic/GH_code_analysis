import csv
import hashlib

reader = list(csv.DictReader(open("./output/participants.csv", 'r')))

row_list = []

for row in reader:
    m = hashlib.md5()
    s = "".join([row['name'], row['login'], row['email']])
    m.update(s)
    row['id'] = m.hexdigest()
    row_list.append(row)

row_list[0]

len(reader)

len(row_list)

fieldnames = [
    'id',
    'name',
    'first_name',
    'last_name',
    'login',
    'email',
    'company',
    'in_coalition',
    'location',
    'state',
    'country',
    'location_x',
    'location_y',
    'in_california',
    'in_usa',
    'contributions',
    'avatar_url',
    'at_san_diego_2016_training'
]

writer = csv.DictWriter(open("./output/participants.csv", 'w'), fieldnames=fieldnames)

writer.writeheader()

writer.writerows(row_list)

