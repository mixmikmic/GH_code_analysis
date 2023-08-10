from scidbpy import connect
sdb = connect('http://localhost:8080')

lenght = len(dir(sdb.arrays))
lenght

from random import randint
arrayName = dir(sdb.arrays)[randint(0, lenght)]
arrayName

myArray = sdb.iquery('scan('+arrayName+')', fetch=True, as_dataframe=True)

import matplotlib.pyplot as plt
simpleArray = myArray["x"];
simpleArray.plot()
plt.show()

simpleArray



