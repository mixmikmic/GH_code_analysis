from collections import namedtuple

Rectangle = namedtuple('Rectangle',['width','height'],verbose=False)

print type(Rectangle)

r1 = Rectangle(10,20)
print type(r1)

print r1
print r1._asdict()
rect = [(10,20),(1,2),(5,8)]
print map(Rectangle._make,rect)
print r1._fields
print r1._replace(width=12)
print getattr(r1,'height')

import csv
country_record = namedtuple('countryRecord',['country_name','country_code','country_iso_code'],verbose=False)
country_tupple = map(country_record._make,csv.reader(open("country_codes.csv","rb")))
for c in country_tupple[0:3]:
    print c
    print c.country_name,c.country_iso_code

# random access - go for list
# accessing at the ends - go for deque

from IPython import display
display.Image("basicdeque.png")

from collections import deque

numberList = range(1,100000)
myque = deque(numberList)
print type(myque)

def get_element(deque_object):
    a = deque_object.pop()
    deque_object.append(a)
    return a

print get_element(myque)
print len(myque)

newList = [100000,100001,100002]
myque.extend(newList)
print get_element(myque)
print len(myque)

try :
    value = 100001
    myque.remove(value)
except ValueError:
    print "Value " + str(value) + " not in myque"
print get_element(myque)

rect = [(10,10),(20,20),(30,30),(40,40),(50,50)]
my_rect_deque = deque(map(Rectangle._make,rect))
print get_element(my_rect_deque)

for a in range(len(my_rect_deque)):
    x = my_rect_deque.pop()
    print x
    my_rect_deque.appendleft(x)

my_rect_deque.rotate(-2)

#import pandas as pd
#import numpy as np
#df = pd.read_html('https://countrycode.org/')[1]
#df.to_csv('country_codes.csv',index=False,header=False)

from collections import Counter

votes = ['bjp','congress','aap','bjp','bjp','bjp','congress','aap','bjp','bjp','jdal','bjp','congress','aap']

votes_counter = Counter(votes)

print votes_counter

print list(votes_counter.elements())

votes_counter.most_common(3)

invalid_votes = ['bjp','aap','congress','newparty']

total_votes = Counter(votes)
total_votes.subtract(invalid_votes)
print total_votes

total_votes.update(['aap','bjp','others'])
print total_votes

from collections import OrderedDict

new_ordered_dict = OrderedDict()

new_ordered_dict.update({'Root':47})

new_ordered_dict.update({'Parent1':40})

new_ordered_dict.update({'Parent2':38})

new_ordered_dict

xx = OrderedDict([('a',10),('b',20),('c',40)])
Person = namedtuple('Person',['name','age'],verbose=False)

persons = map(Person._make,[('jonh',25),('mary',16),('david',20),('karmen',5)])

persons_od = OrderedDict(sorted(persons,key=lambda x:x.age))

persons_od.update({'Tiger':60})
print persons_od

persons_od.pop('karmen')
print persons_od

print persons_od.popitem()

print persons_od.popitem(last=False)

for x in reversed(persons_od):
    print x

from collections import defaultdict

my_list = [('bjp','Modi'),('aap','Kejariwal'),('bjp','Rajnath'),('congress','Sonia'),('congress','Rahul'),('aap','Manish')]
my_def_dict = defaultdict(list)
for k,v in my_list:
    my_def_dict[k].append(v)
print my_def_dict

bjp_party_people = my_def_dict.get('bjp')
print bjp_party_people

ramayan_characters = defaultdict(Person)

ramayan_characters['Ram'] = Person('Ram',35)
ramayan_characters['Lakshman'] = Person('Lakshman',30)
ramayan_characters['Sita'] = Person('Sita',28)
print ramayan_characters

ramayan_characters['Lakshman'] = ramayan_characters['Lakshman']._replace(age=29)
print ramayan_characters['Lakshman']



