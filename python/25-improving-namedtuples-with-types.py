# "regular" namedtuples
from collections import namedtuple

# let's create an Employee object
Employee = namedtuple('Employee',
                      'name age title department job_description')

bob = Employee('Bob Smith', 
               35, 
               'Senior Software Engineer', 
               'Technology',
               ['mentor junior developers',
                'fix production issues',
                'understand deployment pipeline'])

bob

attribute_to_examine = bob.name
print(attribute_to_examine)
print(type(attribute_to_examine))

attribute_to_examine = bob.age
print(attribute_to_examine)
print(type(attribute_to_examine))

attribute_to_examine = bob.title
print(attribute_to_examine)
print(type(attribute_to_examine))

attribute_to_examine = bob.department
print(attribute_to_examine)
print(type(attribute_to_examine))

attribute_to_examine = bob.job_description
print(attribute_to_examine)
print(type(attribute_to_examine))

print(bob[0])

## versus

print(bob.name)

from typing import List, NamedTuple

class EmployeeImproved(NamedTuple):
    name: str
    age: int
    title: str
    department: str
    job_description: List

emma = EmployeeImproved('Emma Johnson',
                        28,
                        'Front-end Developer',
                        'Technology',
                        ['build React components',
                         'test front-end using Selenium',
                         'mentor junior developers'])

emma

attribute_to_examine = emma.name
print(attribute_to_examine)
print(type(attribute_to_examine))

attribute_to_examine = emma.age
print(attribute_to_examine)
print(type(attribute_to_examine))

attribute_to_examine = emma.title
print(attribute_to_examine)
print(type(attribute_to_examine))

attribute_to_examine = emma.department
print(attribute_to_examine)
print(type(attribute_to_examine))

attribute_to_examine = emma.job_description
print(attribute_to_examine)
print(type(attribute_to_examine))

emma[0]

