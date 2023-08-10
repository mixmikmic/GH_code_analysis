import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
bus_df = pd.read_csv('data_acquisition/combined/bus.csv')
metro_df = pd.read_csv('data_acquisition/combined/metro.csv')
taxi_df = pd.read_excel('data_acquisition/combined/taxi.xlsx')
uber_df = pd.read_csv('data_acquisition/combined/uber.csv')

bus_df.columns=['Date','Bus']
bus_df['Date'] = pd.to_datetime(bus_df['Date'],format='%Y%m')
bus_df.dtypes
bus_df.head()

metro_df = metro_df[['Date', 'ROTP', 'RailReliability', 'MetroAccessOTP',
       'EscalatorAvail', 'ElevatorAvail', 'TotalInjuries', 'Crime',
       'Ridership']]
metro_df['Date'] = pd.to_datetime(metro_df['Date'])
metro_ride = metro_df[['Date','Ridership']]
metro_ride.head()

taxi_df.columns=['Date','Taxi']
taxi_df['Date'] = pd.to_datetime(taxi_df['Date'],format='%Y%m')
taxi_df.head()

taxi_df

uber_df.columns = ['Date','Uber']
uber_df['Date'] = pd.to_datetime(uber_df['Date'],format='%Y%m')
uber_df.head()

print(bus_df.shape)
print(metro_df.shape)
print(taxi_df.shape)
print(uber_df.shape)

combined_df = pd.merge(bus_df,metro_ride,on='Date',how='outer')
combined_df = pd.merge(combined_df,taxi_df,on='Date',how='outer')
combined_df = pd.merge(combined_df,uber_df,on='Date',how='outer')
combined_df = combined_df.sort_values('Date').set_index('Date')
combined_df

combined_df.to_csv('combined.csv')

combined_df.dtypes

plt.style.use('fivethirtyeight')
f, ax = plt.subplots(figsize=(15,10))
combined_df.plot(ax=ax,linewidth=3)
plt.show()

metro_df = metro_df[['Date','ROTP', 'RailReliability', 'MetroAccessOTP', 'EscalatorAvail',
       'ElevatorAvail', 'TotalInjuries', 'Crime',]]
metro_df = metro_df.set_index('Date')

f, ax2 = plt.subplots(figsize=(15,10))
metro_df['ROTP'].plot(ax=ax2,linewidth=3)
plt.suptitle('ROTP')
plt.show()

metro_df.head()

#to check if there is a realtion between variables
combined_df.plot(x='Ridership',y='Taxi',kind='scatter')
combined_df.plot(x='Ridership',y='Uber',kind='scatter')
combined_df.plot(x='Ridership',y='Bus',kind='scatter')

plt.show()

plt.style.use('fivethirtyeight')
f, ax = plt.subplots(figsize=(15,10))
combined_subset.plot(ax=ax,linewidth=3)
plt.show()

# Dropping NA's is required to use numpy's polyfit
combined_subset = combined_df.dropna(subset=['Bus', 'Ridership','Taxi','Uber'])
#ploting trend line for taxi rides and metro ridership
y=combined_subset['Taxi']
x=combined_subset['Ridership']
plt.scatter(x,y)
plt.plot(np.unique(x),
         np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),
         color='blue')
plt.xlabel('Metro Ridership')
plt.ylabel('Taxi Rides')
plt.title('OLS(Ordinary least Square) relationship between Taxi Rides and Metro Ridership')
plt.show()



#Regression model between Taxi Rides and Metro Ridership
Taxi_model = smf.OLS(y, x).fit()
# make the predictions by the model
#predictions = model.predict(x) 

# Print out the statistics
taxi_model_summary=Taxi_model.summary()
taxi_model_summary

Taxi_Regression_detail={'R2':Taxi_model.rsquared,
'Adjus R2':Taxi_model.rsquared_adj,
'Coeficent':Taxi_model.params,
            'P-values':Taxi_model.pvalues,
            'F-Statistics':Taxi_model.fvalue}
Taxi_Regression_detail

#predicting by the model using the ridership as observation 
Taxi_predicted= Taxi_model.predict(combined_subset["Ridership"]).round(0)
print(Taxi_predicted)
#checking the difference between the observed and the predicted
Taxi_residual=combined_subset['Taxi']-Taxi_predicted

plt.scatter(combined_subset['Ridership'], Taxi_predicted, alpha=0.5, label='predicted')

# Plot observed values

plt.scatter(combined_subset['Ridership'], combined_subset['Taxi'], alpha=0.5, label='observed')

plt.legend()
plt.title('OLS predicted values')
plt.xlabel('Metro Ridership')
plt.ylabel('Taxi Rides')
plt.show()

plt.scatter(Taxi_predicted,Taxi_residual, alpha=0.5, label='predicted')

# Plot observed values

#plt.scatter(combined_subset['Ridership'], combined_subset['Taxi'], alpha=0.5, label='observed')

plt.legend()
plt.title('OLS predicted values')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.show()

#ploting trend line for Bus rides and metro ridership
Y=combined_subset['Bus']
x=combined_subset['Ridership']
plt.scatter(x,Y)
plt.plot(np.unique(x),
         np.poly1d(np.polyfit(x, Y, 1))(np.unique(x)),
         color='blue')
plt.xlabel('Metro Ridership')
plt.ylabel('Bus Rides')
plt.title('OLS(Ordinary least Square) relationship between Bus Rides and Metro Ridership')
plt.show()

#Regression model between Bus Rides and Metro Ridership
Bus_model = smf.OLS(Y, x).fit()
# make the predictions by the model
#predictions = model.predict(Y) 

# Print out the statistics
Bus_model_summary=Bus_model.summary()
Bus_model_summary

Bus_Regression_detail={'R2':Bus_model.rsquared,
'Adjus R2':Bus_model.rsquared_adj,
'Coeficent':Bus_model.params,
            'P-values':Bus_model.pvalues,
            'F-Statistics':Bus_model.fvalue}
Bus_Regression_detail

#using the ridership as observation 
Bus_predicted= Bus_model.predict(combined_subset["Ridership"]).round(0)
print(Bus_predicted)
#checking the difference between the observed and the predicted
Bus_residual=combined_subset['Bus']-Bus_predicted

plt.scatter(combined_subset['Ridership'], Bus_predicted, alpha=0.5, label='predicted')

# Plot observed values

plt.scatter(combined_subset['Ridership'], combined_subset['Bus'], alpha=0.5, label='observed')

plt.legend()
plt.title('OLS predicted values')
plt.xlabel('Metro Ridership')
plt.ylabel('Bus Rides')
plt.show()

plt.scatter(Bus_predicted,Bus_residual, alpha=0.5, label='predicted')

# Plot observed values

#plt.scatter(combined_subset['Ridership'], combined_subset['Taxi'], alpha=0.5, label='observed')

plt.legend()
plt.title('Predicted Vs Residual values')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.show()

#ploting trend line for uber rides and metro ridership
Yu=combined_subset['Uber']
x=combined_subset['Ridership']
plt.scatter(x,y)
plt.plot(np.unique(x),
         np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),
         color='blue')
plt.xlabel('Metro Ridership')
plt.ylabel('Uber Rides')
plt.title('OLS(Ordinary least Square) relationship between Uber Rides and Metro Ridership')
plt.show()

#Regression model between Uber Rides and Metro Ridership
Uber_model = smf.OLS(Yu, x).fit()
# make the predictions by the model
#predictions = model.predict(Y) 

# Print out the statistics
Uber_model_summary=Uber_model.summary()
Uber_model_summary

Uber_Regression_detail={'R2':Uber_model.rsquared,
'Adjus R2':Uber_model.rsquared_adj,
'Coeficent':Uber_model.params,
            'P-values':Uber_model.pvalues,
            'F-Statistics':Uber_model.fvalue}
Uber_Regression_detail

#predicting by the model using the ridership as observation 
Uber_predicted= Uber_model.predict(combined_subset["Ridership"]).round(0)
print(Uber_predicted)
#checking the difference between the observed and the predicted
Uber_residual=combined_subset['Uber']-Uber_predicted

plt.scatter(combined_subset['Ridership'], Uber_predicted, alpha=0.5, label='predicted')

# Plot observed values

plt.scatter(combined_subset['Ridership'], combined_subset['Uber'], alpha=0.5, label='observed')

plt.legend()
plt.title('OLS predicted values')
plt.xlabel('Metro Ridership')
plt.ylabel('Uber Rides')
plt.show()

plt.scatter(Uber_predicted,Uber_residual, alpha=0.5, label='predicted')

# Plot observed values

#plt.scatter(combined_subset['Ridership'], combined_subset['Taxi'], alpha=0.5, label='observed')

plt.legend()
plt.title('Predicted Vs Residual values')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.show()

#Adding a constant column
combined_df['Const']=1
# Dropping NA's is required to use numpy's polyfit
combined_subset_cons = combined_df.dropna(subset=['Bus', 'Ridership','Taxi','Uber','Const'])

Taxi_model_cons = smf.OLS(endog=combined_subset_cons['Taxi'], exog=combined_subset_cons[['Const', 'Ridership']]).fit()
Taxi_model_cons_summary=Taxi_model_cons.summary()
Taxi_model_cons_summary

