# Import our libraries
import pandas as pd

import plotly.offline as offline
import plotly.graph_objs as go

# We're using json this time!
import json

# Run this in offline mode
offline.init_notebook_mode()

# Run the transformations from before to make sure everyone is 
# at the same place.
url = "https://data.delaware.gov/api/views/bxyv-7mgn/rows.csv?accessType=DOWNLOAD"
waterfowl_df = pd.read_csv(url)
waterfowl_df_january = waterfowl_df[waterfowl_df['Month']=='January']
waterfowl_df_january_sub = waterfowl_df_january[waterfowl_df_january['Time Period']!='Late']

waterfowl_df_january_sub_by_year = waterfowl_df_january_sub.groupby('Year').sum()

years = [str(year) for year in waterfowl_df_january_sub_by_year.index]

# Now let's plot the top three.
# This is creating the same chart as before, but options to go.Scatter each have their own line and 
# the layout is being created from a dictionary (our first step toward displaying in Javascript!)

bird_names = ['Canada Goose', 'American Black Duck', 'Mallard']

data = []

for bird_name in bird_names:
    
    single_bird = waterfowl_df_january_sub[['Year', bird_name]].groupby('Year').sum()

    bird_counts = [int(total) for total in single_bird[bird_name]]
    
    # Cheat and re-usse the years variable from before
    data.append(
        go.Scatter(
            x=years, 
            y=bird_counts, 
            mode="markers+lines", 
            name=bird_name
        )
    )

layout_dict = {
    'title': "Top three birds", 
    'xaxis':{'title':'Year'}, 
    'yaxis':{'title':'Number counted'}
}

layout=go.Layout(layout_dict)


figure=go.Figure(data=data,layout=layout)

offline.iplot(figure, filename='top_three')

# ***** This cell requires you to fill something in! *****

# Edit this function to change the style of the chart!

def return_data_dictionary(years, bird_counts, bird_name):
    return {
        # Add to these options to make your own chart
        
        'type':'scatter',
        'x':years, 
        'y':bird_counts, 
        'mode':"markers+lines", 
        'name':bird_name
    }

bird_names = ['Canada Goose', 'American Black Duck', 'Mallard']

# We'll use this variable to create our Javascript graph, but first we'll use it create our Python one!
pre_json_dictionary = {
    'data': []
}

for bird_name in bird_names:
    
    single_bird = waterfowl_df_january_sub[['Year', bird_name]].groupby('Year').sum()

    bird_counts = [int(total) for total in single_bird[bird_name]]
    
    # Append our dictionary to the data list
    pre_json_dictionary['data'].append(return_data_dictionary(years, bird_counts, bird_name))


pre_json_dictionary['layout'] = {
    'title': '{} from {} to {}'.format(', '.join(bird_names), years[0], years[-1]), 
    'xaxis':{'title':'Year'}, 
    'yaxis':{'title':'Number counted'}
}

figure=go.Figure(data=pre_json_dictionary['data'],layout=pre_json_dictionary['layout'])

offline.iplot(figure, filename='top_three')

# This is how we'll display the chart in Javascript.

# Copy the output below, from first '{' to the last '}' 

print(json.dumps(pre_json_dictionary))



