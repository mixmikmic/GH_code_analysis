import pandas as pd
df_temp = pd.read_csv('../data/airquality.csv', 
                      usecols = ["Ozone", "Solar.R", "Wind", "Temp", "Month", "Day"])

# We exclude the first column (= index) because we don't need it.
# To do that, just specify the columns of interest in usecols

#Let's add a year column
df_temp["Year"] = "1973"

# Let's inspect the first few elements
df_temp.head()

# It would be useful to create a date column (better plotting)
df_temp["Date"] = pd.to_datetime(df_temp["Year"] 
                                 + df_temp["Month"].astype(str) 
                                 + df_temp["Day"].astype(str) , format = "%Y%m%d")

# Let's inspect the first few elements again
df_temp.head()

# Check each column's data type
print df_temp.dtypes

# Let's see how to do that with pandas

# Get the data to interpolate
ozone = df_temp["Ozone"].values
solar = df_temp["Solar.R"].values
# Get a series of timestamps
timestamps = pd.to_datetime(df_temp["Date"].values)

# Create a new Series with the timestamp as index 
#(or we could have set the index as timestamp in our df_temp dataframe)
s_ozone = pd.Series(ozone, index=timestamps)
s_solar = pd.Series(solar, index=timestamps)

oz_interp = s_ozone.interpolate(method = "time")
sol_interp = s_solar.interpolate(method = "time")

df_temp["Ozone_interp"] = oz_interp.values
df_temp["Solar.R_interp"] = sol_interp.values

df_temp.head(n=10) # use 10 rows to see how interpolation works

#Let's plot the data
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec

get_ipython().magic('matplotlib inline')
fig = plt.figure(figsize=(15, 15))
gs = GridSpec(2, 2, bottom=0.18, left=0.18, right=0.88)

axOz = fig.add_subplot(gs[0])
axSol = fig.add_subplot(gs[1])
axWind = fig.add_subplot(gs[2])
axTemp = fig.add_subplot(gs[3])

# Get time axis
x_date = df_temp["Date"].values

# Get the y values 
y_oz = df_temp["Ozone_interp"].values
y_sol = df_temp["Solar.R_interp"].values
y_temp = df_temp["Temp"].values
y_wind = df_temp["Wind"].values


# Plot
axOz.plot(x_date, y_oz, label = "Ozone in ppm", linewidth=1.5)
axSol.plot(x_date, y_sol, label = "Solar Radiation", linewidth=1.5)
axWind.plot(x_date, y_temp, label = "Temperature in F", linewidth=1.5)
axTemp.plot(x_date, y_wind, label = "Wind", linewidth=1.5)


#####################
# Figure cosmetics
#####################

# Axis labels, legend and formatting
for ax in [axOz, axSol, axWind, axTemp]:
    ax.set_xlabel("time", fontsize=22)
    ax.legend(loc="best", fontsize=22)
    
# improve plot layout
gs.tight_layout(fig, h_pad=3)

#plt.savefig("visual.png") #uncomment to save plot
plt.show() 

# Clear plotting
plt.clf()
plt.close()

#Let's plot the data
fig = plt.figure(figsize=(15, 15))
gs = GridSpec(2, 1, bottom=0.18, left=0.18, right=0.88)

ax7 = fig.add_subplot(gs[0])
ax30 = fig.add_subplot(gs[1])

# Get time axis
x_date = df_temp["Date"].values

# Get the y values 

r_sol7 = df_temp["Ozone_interp"].rolling(window=7).corr(df_temp["Solar.R_interp"])
r_sol30 = df_temp["Ozone_interp"].rolling(window=30).corr(df_temp["Solar.R_interp"])

r_wind7 = df_temp["Ozone_interp"].rolling(window=7).corr(df_temp["Wind"])
r_wind30 = df_temp["Ozone_interp"].rolling(window=30).corr(df_temp["Wind"])

# On the first few points, no correlation can be computed (because there are not enough previous points)
# This gives a NaN value
# We arbitrarily set this to 0.
r_sol7 = r_sol7.fillna(0)
r_sol30 = r_sol30.fillna(0)
r_wind7 = r_wind7.fillna(0)
r_wind30 = r_wind30.fillna(0)

# Plot
ax7.plot(x_date, r_sol7, label = "7d corr Ozone/Solar", linewidth=1.5, color = "r")
ax7.plot(x_date, r_wind7, label = "7d corr Ozone/Wind", linewidth=1.5, color = "k")
ax30.plot(x_date, r_sol30, label = "30d corr Ozone/Solar", linewidth=1.5, color = "r")
ax30.plot(x_date, r_wind30, label = "30d corr Ozone/Wind", linewidth=1.5, color = "k")

#####################
# Figure cosmetics
#####################

# Axis labels, legend and formatting
for ax in [ax7, ax30]:
    ax.set_xlabel("time", fontsize=22)
    ax.set_ylabel("Correlation coefficient", fontsize=22)
    ax.legend(loc="best", fontsize=22)

# improve plot layout
gs.tight_layout(fig, h_pad=5)

#plt.savefig("visual2.png")
plt.show() #uncomment to plot

