d = {}

type(d)

d['first'] = 1

d

len(d)

d['second'] = 2

d

len(d)

d['third'] = 'wolf'

d

d.keys()

d.values()

d.items()

data = d.items()

data

d = dict(data) #list to dict

d

dow = "mon", "tue", "wed", "thu", "fri", "sat", "sun"

dow

ids = range(len(dow))

ids

dow

zip(dow, ids)

dow_id = dict(zip(dow, ids))

dow_id

dow_di = dict(zip(ids, dow))
dow_di

'mon' in dow_id

dow_id.keys()

dow_id.values()

dow_id.items()

dow_id['sun'] = -4

dow_id

e = {'bono': 'humble single', 'edge': 'the sound'}

e

dow_id

get_ipython().run_line_magic('pinfo', 'dow_id.update')

dow_id.update(e)

dow_id

dir(dow_id)

l = range(5)

l

s = set(l)

s

type(s)

b = [0, 1, 0, 2, 2, 3 , 6]

b

sb = set(b)
sb

c = [ 6, 2, 2, -6, 4]
set(c)

s = 'My name is joe the plumber'
ss = set(s)
ss

s = set(s.split())

s

s = set([1,2,3,4])
s

s.add(5)

s

s.add(1)

s

b = [4,5,6,7]

b

sb = set(b)

sb

s.update(sb)

s

group1 = set(range(1,6))
group2 = set(range(2,7))

group1

group2

group1.intersection(group2)

group1.union(group2)

s1 = set(range(3))
s2 = set(range(10))

s1

s2

s1.issubset(s2)

s2.issuperset(s1)

s1.issuperset(s2)

s2.symmetric_difference(s1)

s1.symmetric_difference(s2)

s2.difference(s1)

s1.difference(s2)

s1

s1.remove(1)

s1

s1.remove(10)

s1.discard(10)

s1

s1.discard(2)

s1

s4 = s1.discard(2)
s4

s = set([1,2,3,'hi'])

s





