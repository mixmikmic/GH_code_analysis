get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean

# stats60 specific
from code import monty_hall
figsize = (8,8)

monty_hall.examples['noswitch']

monty_hall.examples['noswitch'].sample_space

monty_hall.examples['noswitch'].trial()
monty_hall.examples['noswitch']

monty_hall.examples['noswitch'].sample(5)

mean([r[0] == r[3] for r in monty_hall.examples['noswitch'].sample(5000)])

monty_hall.examples['noswitch'].mass_function

monty_hall.examples['switch'].trial()
monty_hall.examples['switch']

mean([r[0] == r[3] for r in monty_hall.examples['switch'].sample(5000)])

monty_hall.examples['switch'].mass_function

monty_hall.examples['switch_match'].trial()
monty_hall.examples['switch_match']

mean([r[0] == r[3] for r in monty_hall.examples['switch_match'].sample(5000)])

monty_hall.examples['switch_match'].mass_function

monty_hall.examples['switch_nomatch'].trial()
monty_hall.examples['switch_nomatch']

mean([r[0] == r[3] for r in monty_hall.examples['switch_nomatch'].sample(5000)])

monty_hall.examples['switch_nomatch'].mass_function

