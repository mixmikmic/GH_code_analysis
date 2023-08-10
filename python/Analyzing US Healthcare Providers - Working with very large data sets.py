import graphlab
import seaborn as sns

get_ipython().magic('matplotlib inline')
graphlab.canvas.set_target('ipynb')

#Run this cell if you do not have a copy of the 'providers_sframe' SFrame file -- WARNING it will take ~10mins to load!
#Complete the path to your copy of the providers_data.csv
url = 'insert\path\to\providers_data.csv' 

#This loads the csv to the graphlab.SFrame object
data = graphlab.SFrame.read_csv(url)

#This saves the SFrame
data.save('providers_sframe')

#Run this cell if you have a copy of the 'providers_sframe' SFrame file. It must be in the same folder as this notebook
data = graphlab.SFrame('providers_sframe')

data.head()

data.shape

# This is a list of the most useful columns for our analysis
cols_of_interest =['NPI',
 'Entity_Type',
 'Provider_Organization_Name',
 'Provider_First_Line_Business_Practice_Location_Address',
 'Provider_Second_Line_Business_Practice_Location_Address',
 'Provider_Business_Practice_Location_Address_City_Name',
 'Provider_Business_Practice_Location_Address_State_Code',
 'Provider_Business_Practice_Location_Address_Postal_Code',
 'Provider_Business_Practice_Location_Address_Country_Code',
 'NPI_Deactivation_Date',
 'Healthcare_Provider_Taxonomy_Code_1',
 'Is_Sole_Proprietor'
]

data = data[cols_of_interest]

data.shape

data.rename({
 'Provider_Organization_Name':'Name',
 'Provider_First_Line_Business_Practice_Location_Address': 'First_Line',
 'Provider_Second_Line_Business_Practice_Location_Address': 'Second_Line',
 'Provider_Business_Practice_Location_Address_City_Name':'City',
 'Provider_Business_Practice_Location_Address_State_Code': 'State_Code',
 'Provider_Business_Practice_Location_Address_Postal_Code': 'Zip',
 'Provider_Business_Practice_Location_Address_Country_Code': 'Country',
})

data['Entity_Type'].show()

# This comand  selects only the rows with the value 'Individual' in the 'Entity_Type' column
data = data[data['Entity_Type']=='Individual']
data

data.shape

# This command evaluates each cell in the 'NPI_Deactivation_Date' column. If a date is present then that row is dropped. 
data = data[data['NPI_Deactivation_Date'].apply(lambda x: len(x)==0)] 
# And lets drop the column now we have finshed with it
data.remove_column('NPI_Deactivation_Date')

data.shape

# SFrames can also read from urls
taxonomy = graphlab.SFrame('http://www.nucc.org/images/stories/CSV/nucc_taxonomy_160.csv')

#Lets take a look at our new taxonomy table...
taxonomy.head()

# First we strip out the columns we dont need
taxonomy = taxonomy[taxonomy.column_names()[:4]]

#Now we make the join of taxonomy to data
data = data.join(taxonomy, on={'Healthcare_Provider_Taxonomy_Code_1':'Code'}, how='left')

data.head()

print data[(data['Classification']=='Dentist' )& (data['State_Code']=='NY') & (data['City']=='KINGS PARK')]



specialities = data.groupby(key_columns = 'Classification', operations = {'count': graphlab.aggregate.COUNT('NPI')} )

specialities.sort('count', ascending = False)

top_20_specialities = specialities.sort('count', ascending = False)[:20]

# Lets plot this to see how it looks
sns.barplot(x='count', y = 'Classification', data = top_20_specialities.to_dataframe())

state_specialities = data.groupby(key_columns = ['Classification', 'State_Code'], 
                                      operations = {'count':  graphlab.aggregate.COUNT('NPI')} )

# And lets take a look at what that data looks like...
state_specialities.head()

state_specialities['State_Code']

# Lets make an array of states we want
states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA',
          'KS', 'KY', 'LA', 'ME', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 
          'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 
          'WV', 'WI', 'WY']

# Here we build a new dataframe by sequntially filtering on each element in the states array 
for index, state in enumerate(states):
    
    if index == 0:
        new_df = state_specialities[state_specialities['State_Code']==state]
    else :    
        new_df = new_df.append(state_specialities[state_specialities['State_Code']==state])
    
# And just update our original dataframe        
state_specialities= new_df

# Take a quick look
state_specialities

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
def create_map (data):
    ax = plt.axes([0, 0, 1, 1], projection=ccrs.LambertConformal())
    ax.set_extent([-125, -66.5, 20, 50], ccrs.Geodetic())
    states_shp = shpreader.natural_earth(resolution='110m',
                                         category='cultural',
                                         name='admin_1_states_provinces_lakes_shp')
    ax.background_patch.set_visible(False)
    ax.outline_patch.set_visible(False)
    title = data['Classification'][0]
    if 'percentage' in data.column_names():
        maximum = float(data['percentage'].max())
        plt.title('Comparative percentages of %s proffessionals by state' % title)
    else:
        maximum = float(data['count'].max())
        plt.title('Number of %s proffessionals by state' % title)
        
    states = shpreader.Reader(states_shp)
    for state in states.records():
        s = state.geometry
        name = state.attributes['postal']
        if 'percentage' in data.column_names():
            alpha = (data[data['State_Code']==name]['percentage'][0])/maximum
        else:
            alpha = (data[data['State_Code']==name]['count'][0])/maximum
        # pick a default color for the land with a black outline
        facecolor = 'red'
        edgecolor = 'black'
        ax.add_geometries([s], ccrs.PlateCarree(),
                               facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)

    plt.show()

create_map(state_specialities[state_specialities['Classification'] == 'Counselor'])

specialities.sort('count', ascending = False)[:50].print_rows(num_rows =50)

# This cell calculates the total number of practioners by state 
percentage_df= state_specialities.groupby(key_columns='State_Code', 
                                    operations ={'TOTAL': graphlab.aggregate.SUM('count')})

#Lets take a look
percentage_df

# Here we join the two tables and calculate the percentage
state_specialities= state_specialities.join(percentage_df, how='left')
state_specialities['percentage']= state_specialities.apply(lambda x: float(x['count']/float(x['TOTAL'])*100 ))
state_specialities

create_map(state_specialities[state_specialities['Classification'] == 'Counselor'])

create_map(state_specialities[state_specialities['Classification'] == 'Chiropractor'])



