import json

input = '''[
  { "id" : "01",
    "status" : "Instructor",
    "name" : "Hrant"
  } ,
  { "id" : "02",
    "status" : "Student",
    "name" : "Jack"
  }
]'''

data = json.loads(input)

type(data)

print 'User count:', len(data)

print(data)

from pprint import pprint

pprint(data)

for item in data:
    print 'Name: ', item['name']
    print 'Id: ', item['id']
    print 'Status: ', item['status'], "\n"

