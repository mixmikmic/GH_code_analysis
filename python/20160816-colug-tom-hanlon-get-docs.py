# !pip install fake-factory ;# not pip install faker

from faker import Factory
import csv
import sys

def foo(filename, n):
    with open(filename, 'w') as f:
        for _ in range(n):
            fake = Factory.create()
            name = fake.profile()['name']
            address_raw = fake.profile()['address']
            address = address_raw.replace('\n', ', ')
            birthdate = fake.profile()['birthdate']
            phone = fake.phone_number()
            writer = csv.writer(f)
            writer.writerow((name, address, birthdate, phone))

filename = 'data.csv'
n = 3
foo(filename, n)
print(open(filename).read())

help(Factory)

import faker
help(faker)

help(faker.factory)

help(Factory.create().profile)
# There is a docstring, but it is incomplete.

print(Factory.create().profile.__doc__)

fake = Factory.create().profile()

help(fake)

fake.keys()

fake

