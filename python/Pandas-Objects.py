import numpy as np
import pandas as pd

data = pd.Series([0.25, 0.5, 0.75, 1.0])
data

data.values

data.index

data[1]

data[1:3]

data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
data

data['b']

data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=[2, 5, 3, 7])
data

data[5]

population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
population

population['California']

population['California':'Illinois']

pd.Series([2, 4, 6])

pd.Series(5, index=[100, 200, 300])

pd.Series({2:'a', 1:'b', 3:'c'})

pd.Series({2:'a', 1:'b', 3:'c'}, index=[3, 2])

area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)
area

states = pd.DataFrame({'population': population,
                       'area': area})
states

states.index

states.columns

states['area']

pd.DataFrame(population, columns=['population'])

data = [{'a': i, 'b': 2 * i}
        for i in range(3)]
pd.DataFrame(data)

pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])

pd.DataFrame({'population': population, 'area': area})

pd.DataFrame(np.random.rand(3, 2),
             columns=['foo', 'bar'],
             index=['a', 'b', 'c'])

A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')])
A

pd.DataFrame(A)

ind = pd.Index([2, 3, 5, 7, 11])
ind

ind[1]

ind[::2]

print(ind.size, ind.shape, ind.ndim, ind.dtype)

ind[1] = 0

indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])

indA & indB  # intersection

indA | indB  # union

indA ^ indB  # symmetric difference

