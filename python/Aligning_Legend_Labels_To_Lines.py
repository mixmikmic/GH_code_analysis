get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt  
import seaborn as sns
import pandas as pd
sns.set_style("whitegrid")
custom_style = {
            'grid.color': '0.8',
            'grid.linestyle': '--',
            'grid.linewidth': 0.5,
}
sns.set_style(custom_style)

df = pd.read_csv('/home/pybokeh/Dropbox/python/jupyter_notebooks/matplotlib/percent-bachelors-degrees-women-usa.csv',
                index_col=0)

df.head()

def legend_positions(df, y):
    """ Calculate position of labels to the right in plot... """
    positions = {}
    for column in y:    
        # Get y value based on last valid index
        positions[column] = df[column][df[column].last_valid_index()] - 0.1

    def push():
        """ 
        ...by puting them to the last y value and 
        pushing until no overlap 
        """
        collisions = 0
        for column1, value1 in positions.items():
            for column2, value2 in positions.items():
                if column1 != column2:
                    dist = abs(value1-value2)
                    if dist < 2.5:
                        collisions += 1
                        if value1 < value2:
                            positions[column1] -= .1
                            positions[column2] += .1
                        else:
                            positions[column1] += .1
                            positions[column2] -= .1
                        return True
    while True:
        pushed = push()
        if not pushed:
            break
            
    return positions

y = df.columns  # Identify the y columns
positions = legend_positions(df, y)

f, ax = plt.subplots(figsize=(11,8))        
cmap = plt.cm.get_cmap('Paired', len(y))

for i, (column, position) in enumerate(positions.items()):
    
    # Get a color
    color = cmap(float(i)/len(positions))
    # Plot each line separatly so we can be explicit about color
    ax = df.plot(y=column, legend=False, ax=ax, color=color)
    
    # Add the text to the right
    plt.text(
        # Get x value of the specific series column based on last valid index
        df[column].last_valid_index() + 0.1, 
        position, column, fontsize=12, 
        color=color # Same color as line
    ) 
ax.set_ylabel('Female bachelor degrees')
# Add percent signs
ax.set_yticklabels(['{:3.0f}%'.format(x) for x in ax.get_yticks()]) 
sns.despine()

df = pd.read_csv('/home/pybokeh/Dropbox/python/jupyter_notebooks/matplotlib/defect_rate.csv', index_col=0)

df

y = df.columns
positions = legend_positions(df, y)

f, ax = plt.subplots(figsize=(6, 4))        
cmap = plt.cm.get_cmap('Paired', len(y))

for i, (column, position) in enumerate(positions.items()):
    
    # Get a color
    color = cmap(float(i)/len(positions))
    # Plot each line separatly so we can be explicit about color
    ax = df.plot(y=column, legend=False, ax=ax, color=color)
    
    # Add the text to the right
    plt.text(
        # Get x value of the specific series column based on last valid index
        df[column].last_valid_index() + 0.1, 
        position, column, fontsize=12, 
        color=color # Same color as line
    ) 
ax.set_ylabel('Cumulative Defect Rate (%)')
# Add percent signs
ax.set_yticklabels(['{:3.0f}%'.format(x) for x in ax.get_yticks()])
ax.set_title("Component Failures", weight='bold')
sns.despine()

