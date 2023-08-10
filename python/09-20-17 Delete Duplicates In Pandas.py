import pandas as pd

raw_data = {'first_name': ['Jason', 'Jason', 'Jason','Tina', 'Jake', 'Amy'], 
        'last_name': ['Miller', 'Miller', 'Miller','Ali', 'Milner', 'Cooze'], 
        'age': [42, 42, 1111111, 36, 24, 73], 
        'preTestScore': [4, 4, 4, 31, 2, 3],
        'postTestScore': [25, 25, 25, 57, 62, 70]}
df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'preTestScore', 'postTestScore'])
df

df.duplicated()

df.drop_duplicates()

df.drop_duplicates(['first_name'], keep='last')

