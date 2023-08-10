import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import pandas as pd

#Define connection. 
#engine=create_engine('postgresql://username:password@host:port/databasename)
engine=create_engine('postgresql://')

#review table names
table_names=engine.table_names()
print(table_names)

df=pd.read_sql_query('SELECT * from individual_contribution_join_abbreviated', engine)

df.head()

df = df.drop("cmte_id", axis = 1)
df = df.drop("amndt_ind", axis = 1)
df = df.drop("rpt_tp", axis = 1)
df = df.drop("result", axis = 1)
df = df.drop("cand_pty_affliation", axis = 1)
df = df.drop("transaction_pgi", axis = 1)
df = df.drop("entity_tp", axis = 1)
df = df.drop("name", axis = 1)
df = df.drop("sub_id", axis = 1)

df.head()

df.to_csv('fecindividual.csv', sep = ",", index = False)

df = pd.read_csv('fecindividual.csv')

df.head()

df['transaction_amt'] = df['transaction_amt'].astype('int')
df['cand_id'] = df['cand_id'].astype('str')
df['state'] = df['state'].astype('str')
df['transaction_dt'] = df['transaction_dt'].astype('str')

df['transaction_dt'] = df['transaction_dt'].str[3:7]
df.head()

df.loc[df['transaction_dt'] == '2013', 'transaction_dt'] = '2014'
df.head()

df = df[(df.transaction_dt == "2014")]
df = df[df['transaction_amt']> 0]
df.head()

df.to_csv('donationsclean.csv', sep = ',', index = False)

df = df.drop("transaction_dt", axis = 1)
df['cand_id'] = df['cand_id'].str[2:4]
df = df[(df.cand_id != df.state)]
df = df.drop("state", axis = 1)

dfgroup = df.groupby(['cand_id']).transaction_amt.sum()
dfgroup.head()

dfgroup.to_csv("DonationsByDonorState.csv", sep = ',', header = False)

dfgroup = pd.read_csv("DonationsByDonorState.csv")

dfgroup.columns = ['state', 'donations']

dfstates = pd.read_csv('HouseRepState.csv')
dfstates.head()

dfstates.columns = ['state name', 'number of seats', 'state']

df_donor_state = dfgroup.merge(dfstates,on='state')
df_donor_state = df_donor_state.drop("state name", axis = 1)
df_donor_state.columns = ['state', 'donations', 'numberofseats']
df_donor_state = df_donor_state.drop("numberofseats", axis = 1)
df_donor_state.head()

df_donor_state.to_csv("OutofStateDonationsSpentInState.csv", sep = ",", index = False)

dfdonor = pd.read_csv("DonationsByDonorStateFinal.csv")
dfdonor.head()

init_notebook_mode(connected=True)

for col in dfdonor.columns:
    dfdonor[col] = dfdonor[col].astype(str)

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


dfdonor['text'] = dfdonor['state'] + '<br>' +    'Amount Donated '+dfdonor['donations']

    
    
data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = dfdonor['state'],
        z = dfdonor['donations'].astype(float),
        locationmode = 'USA-states',
        text = dfdonor['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Dollars Donated")
        ) ]

layout = dict(
        title = 'Amount Spent in 2014 Congressional Elections By Out Of State of Donors' + '<br>' + '(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
iplot(fig)

df = pd.read_csv('donationsclean.csv')

df = df.drop("transaction_dt", axis = 1)
df['cand_id'] = df['cand_id'].str[2:4]
df = df[(df.cand_id != df.state)]
df = df.drop("cand_id", axis = 1)

dfgroup = df.groupby(['state']).transaction_amt.sum()
dfgroup.head()

dfgroup.to_csv("OutOfStateDonationsByDonor.csv", sep = ',', header = False)

dfgroup = pd.read_csv("OutOfStateDonationsByDonor.csv")

dfgroup.columns = ['state', 'donations']

dfstates = pd.read_csv('HouseRepState.csv')
dfstates.head()

dfstates.columns = ['state name', 'number of seats', 'state']

df_donor_state = dfgroup.merge(dfstates,on='state')
df_donor_state = df_donor_state.drop("state name", axis = 1)
df_donor_state.columns = ['state', 'donations', 'numberofseats']
df_donor_state = df_donor_state.drop("numberofseats", axis = 1)
df_donor_state.head()

df_donor_state.to_csv("OutofStateDonationsFromState.csv", sep = ",", index = False)

dfdonor = pd.read_csv("OutofStateDonationsFromState.csv")
dfdonor.head()

init_notebook_mode(connected=True)

for col in dfdonor.columns:
    dfdonor[col] = dfdonor[col].astype(str)

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


dfdonor['text'] = dfdonor['state'] + '<br>' +    'Amount Donated '+dfdonor['donations']

    
    
data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = dfdonor['state'],
        z = dfdonor['donations'].astype(float),
        locationmode = 'USA-states',
        text = dfdonor['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Dollars Donated")
        ) ]

layout = dict(
        title = 'Donors who sent Money to Out of State 2014 Congressional Elections' + '<br>' + '(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
iplot(fig)

df = pd.read_csv("donationsclean.csv")
df.head()

df['cand_id'] = df['cand_id'].str[2:4]
df.head()

df = df.drop("state", axis = 1)

df.to_csv("DonationsByRaceState.csv", sep = ',', index = False)

df.head()

df = df.drop("transaction_dt", axis = 1)

dfgroup = df.groupby(['cand_id']).transaction_amt.sum()
dfgroup.head()

dfgroup.to_csv("DonationsByRaceState.csv", sep = ',', header = False)

dfgroup = pd.read_csv("DonationsByRaceState.csv", header = None)
dfgroup.head()

dfgroup.columns = ['state', 'donations']

dfstates = pd.read_csv('HouseRepState.csv')

dfstates.columns = ['state name', 'numberofseats', 'state']

df_race_state = dfgroup.merge(dfstates,on='state')
df_race_state.head()

df_race_state['normalized'] = df_race_state['donations']/df_race_state['numberofseats']

df_race_state.head()

df_race_state = df_race_state.drop("numberofseats", axis = 1)
df_race_state = df_race_state.drop("state name", axis = 1)
df_race_state = df_race_state.drop("donations", axis = 1)

df_race_state.to_csv("DonationsByRaceStateFinal.csv", sep = ',', index = False)

init_notebook_mode(connected=True)

df_state = pd.read_csv("DonationsByRaceStateFinal.csv")
df_state.head()

for col in df_state.columns:
    df_state[col] = df_state[col].astype(str)

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


df_state['text'] = df_state['state'] + '<br>' +    'Amount Donated '+df_state['normalized']

    
    
data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df_state['state'],
        z = df_state['normalized'].astype(float),
        locationmode = 'USA-states',
        text = df_state['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Dollars Donated")
        ) ]

layout = dict(
        title = 'Average Amount Spent In Each State in 2014 Congressional Representatives Elections' + '<br>' + '(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
iplot(fig)

