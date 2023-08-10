# Import pandas as pd
import pandas as pd

# Create an example dataframe
data = {'name': ['A', 'B', 'C', 'D', 'E'], 
        'score': [1,2,3,4,5]}
df = pd.DataFrame(data)
df

# Select rows of the dataframe where df.score is greater than 1 and less than 5
df[(df['score'] > 1 ) & (df['score'] < 5)]

