import pandas as pd
import requests
import io

# UCI machine learning has a white wine and red wine quality dataset
red_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
white_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'

# use requests to secure the data
red_raw = requests.get(red_url).content
white_raw = requests.get(white_url).content

# we can use the io package to decode the data and feed into pandas.
# A good discussion on this can be located at: https://stackoverflow.com/questions/32400867/pandas-read-csv-from-url

red_df = pd.read_csv(io.StringIO(red_raw.decode('utf-8')), sep=';')
white_df = pd.read_csv(io.StringIO(white_raw.decode('utf-8')), sep=';')

# assign the type of wine
red_df['Type'] = 'Red'
white_df['Type'] = 'White'

# and finally merge into a single dataset
df = pd.concat([red_df, white_df])

assert df.shape == (6497, 13), 'Incorrect number of rows obtains'

df.head(10)

# we will bin the alcohol content into low, medium, high to further illustrate pandas split/apply/combine functionality
df['alcohol'].describe()

# create the three bins
df['alcoholbins'] = pd.cut(df['alcohol'], 3, labels=['low', 'medium', 'high'])

# get the mean alcohol content for low, medium, high bins
df.groupby('alcoholbins')['alcohol'].mean()

# and lets see how many observations there are under each
df.groupby('alcoholbins')['alcohol'].count()

# you can also iterate over a groupby in a regular for loop
for name, group in df.groupby('alcoholbins'):
    print(name)

# and to see further what is happening, each group assigned to the group variable in the for loop 
# is your actual data for that group. You can see this by peeking into the groups:

for name, group in df.groupby('alcoholbins'):
    print('---------------')
    print('Current group: {}'.format(name))
    print(group.head(2))

# occasionally, you will want to derive new features using aggregate values for a group. In order to assign these
# values back to the original dataframe, we can reset the index and merge back in using the following:

# create a copy of the dataframe
dfcopy = df.copy(deep=True)

# get the counts by group, reset the idnex, and give the column a more meaningful name
alc_counts = df.groupby('alcoholbins')['alcohol'].count().reset_index().rename(columns={'alcohol': 'alcoholcount'})

# merge back in with original data
dfcopy = pd.merge(dfcopy, alc_counts, on='alcoholbins', how='inner')

# and now we have assigned the total observations per alcohol bin to all rows of the dataframe
dfcopy.head()

# to unpack the groupby mean, it is the same as doing the following except for all groups

df[df['alcoholbins'] == 'low'].loc[:, 'alcohol'].mean() # which matches our original groupby value for the low bin

# now what if we were interested in the average alcohol content by alcohol bin and by type of wine?
# simply add the additional groupby variable
df.groupby(['Type', 'alcoholbins'])['alcohol'].mean()

# this is the same as the following
df[(df['Type'] == 'Red') & (df['alcoholbins'] == 'low')].loc[:, 'alcohol'].mean()

# sometimes we may be interested in different types of aggregation methods based on the columns.
# for instance, if we want to find the average chlorides by group by the standard deviation for volatile acidity
# we can use:

res = df.groupby(['Type', 'alcoholbins']).agg({'chlorides': 'mean',
                                        'volatile acidity': 'std'}).reset_index().rename(columns={'chlorides': 'chl_mean',
                                                                                         'volatile acidity': 'acid_std'})

res.head()

# and to merge back in with the original data
dfcopy = pd.merge(dfcopy, res, on=['Type', 'alcoholbins'], how='inner')

dfcopy.head(10)

# we can also retrieve specific groups after a groupby has been applied

groups = df.groupby(['Type', 'alcoholbins']).get_group(('White', 'low'))

groups.head()

def custom_group_func(group):
    """
    perform custom operation on group of data. Here we will tag quality groups defined as average quality
    score above 9
    """
    group_mean = group['quality'].mean()
    if group_mean > 6:
        group['Quals'] = 'High Quality Group'
    else:
        group['Quals'] = 'Regular Quality Group'
    return group.reset_index(drop=True)

res = pd.concat([df.groupby(['Type', 'alcoholbins']).apply(custom_group_func)]).reset_index(drop=True)
res.head()

# So what ar ethe high quality groups?
res[res['Quals'].str.contains('High')][['Type', 'alcoholbins']].drop_duplicates()
# Red Wine with high alcohol
# white wine with medium alcohol
# White wine with high alcohol
# seems as if people really enjoy their high alcohol content

# just out of pure curiousity, we can see the average quality by alcohol content
qual_means_groups = df.groupby('alcoholbins')['quality'].mean()
qual_mean = df['quality'].mean()
std = df['quality'].std()
print("""Average Quality: {}
        \n--------------\nQuality Groups Standard Deviation: {}
        \n--------------\n{}""".format(qual_mean, std, qual_means_groups))

# wines with high alcohol content are nearly a full standard deviation away from the average quality across all wines
# people definitely enjoy their alcohol content :) 

def flag_n_std(group, 
                std_flag=2, 
                pop_mean=None,
                pop_std=None,
                col_of_interest='quality',
                flag='STD_FLAG'):
    """
    flag any groups more than std_flag above the population mean
    
    params
    -------
    group --> group of data being operated on
    std_flag --> number of standard deviations away 
    col_of_interest --> Column we want to flag as being std_flag away from the population mean
    pop_mean --> population mean for column
    pop_std --> population standard deviation
    flag --> value to fill in as our flag
    """
    # get group mean
    group_mean = group.loc[:, col_of_interest].mean()
    # get the threshold
    thresh = std_flag * pop_std
    # get the difference
    diff = group_mean - pop_mean
    # check
    if diff > thresh:
        group['STD_FLAG'] = flag
    else:
        group['STD_FLAG'] = None
    return group.reset_index()

# set some basic parameter values
col_of_interest = 'quality'
pop_mean = df[col_of_interest].mean()
pop_std = df[col_of_interest].std()
flag = 'GREAT_QUALITY'

# apply partial function to dataframe
res = pd.concat([df.groupby(['Type', 'alcoholbins']).apply(flag_n_std, std_flag=0.5,
                                                          pop_mean=pop_mean,
                                                          pop_std=pop_std,
                                                          flag=flag).reset_index(drop=True)])

res[res['STD_FLAG'].notnull()][['Type', 'alcoholbins']].drop_duplicates()
# across both red and white wines, high alcohol content translates into better quality :) 

