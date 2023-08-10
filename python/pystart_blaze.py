from blaze import *
#from blaze import compute
#from blaze import data
#from blaze.utils import example
from blaze import examples

help(blaze)
#help(examples)

js = JSON(example('accounts.json'))
s = symbol('s', discover(js))
compute(s.count(), js)

#jss = JSONLines(example('accounts-streaming.json'))
#compute(s.count(), jss)

t = Data([(1, 'Alice', 100),
        (2, 'Bob', -200),
        (3, 'Charlie', 300),
        (4, 'Denis', 400),
        (5, 'Edith', -500)],
        fields=['id', 'name', 'balance'])
#help(t)
t

iris = Data(example('iris.csv'))
iris



