# convert string to uppercase in Python
'hello'.upper()

import pandas as pd

url = 'http://bit.ly/chiporders'
orders = pd.read_table(url)

orders.head()

# .str is a string method
orders.item_name.str.upper()

# you can overwrite with the following code
orders.item_name = orders.item_name.str.upper()

orders.head()

orders.item_name.str.contains('Chicken').head()

# replacing elements
orders.choice_description.str.replace('[', '').head()

# chain string methods
orders.choice_description.str.replace('[', '').str.replace(']', '').head()

# using regex to simplify the code above
orders.choice_description.str.replace('[\[\]]', '').head()

