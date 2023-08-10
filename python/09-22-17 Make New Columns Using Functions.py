# Import Modules
import pandas as pd

# Exmaple dataframes
raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'], 
        'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'], 
        'name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze', 'Jacon', 'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani', 'Ali'], 
        'preTestScore': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
        'postTestScore': [25, 94, 57, 62, 70, 25, 94, 57, 62, 70, 62, 70]}
df = pd.DataFrame(raw_data, columns = ['regiment', 'company', 'name', 'preTestScore', 'postTestScore'])
df

# Create a function that takes two inguts, pre and post
def pre_post_difference(pre, post):
    # return the difference between post and pre
    return post - pre

# Create a vriable that is the output of the function
df['score_change'] = pre_post_difference(df['preTestScore'],
                                        df['postTestScore'])

# View the dataframe
df

# Create a function that takes one input, x
def score_multipler_2x_and_3x(x):
    # return two things, x multipied by 2 and x multiplied by 3
    return x*2, x*3

# Creat tow things, x multiplied by 2 and x multiplied by 3
df['post_score_x2'], df['post_score_x3'] = zip(*df['postTestScore'].map(score_multipler_2x_and_3x))
df

