import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt

get_ipython().magic('matplotlib inline')

df = pd.read_csv('hungarian_nt_matches.csv')

df.head()

def grouped_histograms(grouped_df):
    num_groups = len(grouped_df.groups)
    fig, axs = plt.subplots(int(num_groups/2), 2, figsize=(10, 5*int(num_groups/2)))
    row_count = 0
    counter = 0
    for t in sorted(grouped_df.groups):
        if num_groups == 2:
            if counter == 0:
                current_axis = axs[0]
            else:
                current_axis = axs[1]
        else:
            if counter % 2 == 0:
                current_axis = axs[row_count][0]
            else:
                current_axis = axs[row_count][1]
        grp = grouped_df.get_group(t)
        grp['outcome'].value_counts().reindex(['W', 'D', 'L']).plot(kind='bar', ax=current_axis)
        current_axis.set_title(t)
        if counter > 0 and counter % 2 == 0:
            row_count += 1
        counter += 1

    plt.show()

by_type = df.groupby(by='match_type')
grouped_histograms(by_type)

by_hoa = df.groupby(by='home_or_away')
grouped_histograms(by_hoa)

by_opponent = df.groupby(by='opponent')
grouped_histograms(by_opponent)

