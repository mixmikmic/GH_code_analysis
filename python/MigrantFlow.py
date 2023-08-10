#!pip3 install ipysankeywidget
#!jupyter nbextension enable --py --sys-prefix ipysankeywidget

import pandas as pd

#Data from ONS: https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/migrationwithintheuk/datasets/matricesofinternalmigrationmovesbetweenlocalauthoritiesandregionsincludingthecountriesofwalesscotlandandnorthernireland

#Read in the CSV file
#If we specify the null character and thousands separator, the flows whould be read in as numerics not strings
df=pd.read_csv("../data/laandregionsquarematrices2015/regionsquarematrix2015.csv",
               skiprows = 8,thousands=',',na_values='-')
df.head()

from ipysankeywidget import SankeyWidget

#The widget requires an edgelist with source, target and value columns
dfm=pd.melt(df,id_vars=['DESTINATION','Region'], var_name='source', value_name='value')
dfm.columns=['DESTINATION','target','source','value']
dfm['target']=dfm['target']+'_'
dfm.head()

#The SankeyWidget function expects a list of dicts, each dict specifying an edge
#Also check how to drop rows where the weight is NA
links=dfm.dropna()[['source','target','value']].to_dict(orient='records')
links[:3]

#Generate and display default styled Sankey diagram
SankeyWidget(value={'links': links},
             width=800, height=800,margins=dict(top=0, bottom=0))

colormap={'E':'#ffcc00','N':'green','S':'blue','W':'red'}
dfm['color']=dfm['source'].apply(lambda x: colormap[x[0]])

links = dfm.dropna()[['source','target','value','color']].to_dict(orient='records')
SankeyWidget(value={'links': links},
             width=800, height=800,margins=dict(top=0, bottom=0))

#Create a data frame with dropped flow between countries
#That is, ignore rows where the country code indication is the same between source and target
#Again, drop the rows with unspecificed flows
dfmb = dfm[dfm['source'].str[0]!=dfm['target'].str[0]].dropna()

links= dfmb[['source','target','value','color']].to_dict(orient='records')
SankeyWidget(value={'links': links}, width=800, height=800,margins=dict(top=0, bottom=0))

countrymap={'E':'England','N':'Northern Ireland','S':'Scotland','W':'Wales'}
dfmb['countrysource']=dfmb['source'].apply(lambda x: countrymap[x[0]])
dfmb['countrytarget']=dfmb['target'].apply(lambda x: countrymap[x[0]]+' ')

#Group the (regional) country-country rows and sum the flows, resetting the table to flat columns
dfmg = dfmb.groupby(['countrysource','countrytarget']).aggregate(sum).reset_index()

#Rename the columns as required by the Sankey plotter
dfmg.columns=['source','target','value']

#And apply colour
dfmg['color']=dfmg['source'].apply(lambda x: colormap[x[0]])
dfmg

links=dfmg.to_dict(orient='records')

s=SankeyWidget(value={'links': links},
             width=800, height=800,margins=dict(top=0, bottom=0,left=150,right=120))
s

#!mkdir -p images
#Save a png version
s.save_png('images/mySankey.png')

#save svg
s.save_svg('images/mySankey.svg')

from IPython.display import SVG, display
display(SVG('images/mySankey.svg'))



