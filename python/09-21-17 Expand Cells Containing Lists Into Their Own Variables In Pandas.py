# Import pandas
import pandas as pd

# create a dataset
raw_data = {'score': [1,2,3], 
        'tags': [['apple','pear','guava'],['truck','car','plane'],['cat','dog','mouse']]}
df = pd.DataFrame(raw_data, columns = ['score', 'tags'])

# view the dataset
df

# expand df.tags into its own dataframe
tags= df['tags'].apply(pd.Series)
tags

# rename each variable is tags
tags = tags.rename(columns = lambda x: 'tag_' + str(x))
tags

# view the tags dataframe
tags

# join the tags dataframe back to the original dataframe
pd.concat([df[:], tags[:]], axis=1)

