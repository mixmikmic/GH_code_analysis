import numpy as np
x = np.array((3, 5, 7))
print(x.sum())
x.min()  

try:
        print(x[0])
except NameError:
       print('x does not exist')

# this can overflow the page
b = "Statistics at UC Berkeley: We are a community engaged in research and education in probability and statistics. In addition to developing fundamental theory and methodology, we are actively"
print(b)

# this code can overflow the page
zoo = {"lion": "Simba", "panda": None, "whale": "Moby", "numAnimals": 3, "bear": "Yogi", "killer whale": "shamu", "bunny:": "bugs"}
print(zoo)

# instead manually break the code
zoo = {"lion": "Simba", "panda": None, "whale": "Moby", 
       "numAnimals": 3, "bear": "Yogi", "killer whale": "shamu", 
       "bunny:": "bugs"}
print(zoo)

# long comments can overflow too
# Statistics at UC Berkeley: We are a community engaged in research and education in probability and statistics. In addition to developing fundamental theory and methodology, we are actively"

# the long output that will appear next in the resulting document (produced from the evaluation of the code above) may wrap to the next line:


