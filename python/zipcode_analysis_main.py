#import Dependencies
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.stats import linregress

#Read data from csv file created by zipcode_build_df.ipynb notebook
#This will be the source for all analysis in this notebook
zipcode_main_df = pd.read_csv('zip_code_data_combined.csv')
del zipcode_main_df["Unnamed: 0"]
zipcode_main_df.head()

#plot complaint count by zipcode

# Create the ticks for bar chart's x axis
plt.figure(figsize=(20,3))
x_axis = np.arange(len(zipcode_main_df["zip_code"]))
tick_locations = [value + 0.4 for value in x_axis]
plt.xticks(tick_locations, zipcode_main_df["zip_code"], rotation="vertical")

plt.bar(x_axis, zipcode_main_df["complaint_count"], alpha=0.5, align="edge", color = "green", edgecolor = "y")

plt.xlabel("Zipcode")
plt.ylabel("Complaint Count")
plt.title("Complaint count by zipcode")

plt.style.use('ggplot')

plt.show()

#complaint count vs population
plt.scatter (zipcode_main_df["total_population"], zipcode_main_df["complaint_count"], color = "green", edgecolor = "g")
                               
plt.xlabel("Population")
plt.ylabel("Complaint count")
plt.title("Complaint count by population")

plt.show()

#calculate per-capita complaint count by zipcode
zipcode_main_df["per_capita_complaint"] = zipcode_main_df["complaint_count"]/zipcode_main_df["total_population"]
zipcode_main_df.head()

#plot per-capita complaint count by zipcode

# Create the ticks for bar chart's x axis
plt.figure(figsize=(20,3))
x_axis = np.arange(len(zipcode_main_df["zip_code"]))
tick_locations = [value + 0.4 for value in x_axis]
plt.xticks(tick_locations, zipcode_main_df["zip_code"], rotation="vertical")

plt.bar(x_axis, zipcode_main_df["per_capita_complaint"], alpha=0.5, align="edge", color = "green", edgecolor = "y")

plt.xlabel("Zipcode")
plt.ylabel("Per-capita Complaint Count")
plt.title("Per-capita Complaint count by zipcode")

plt.show()

# Create data-frame for zipcodes with populations between 20K-30K
# This will be used later for analysis of zip-codes with similar population
zipcode_20k_to_30k = zipcode_main_df[zipcode_main_df["total_population"].between(20000, 30000, inclusive=True)]

#complaint count vs income
plt.scatter (zipcode_main_df["median_household_income($)"], zipcode_main_df["complaint_count"])
plt.xlabel("Median household income($)")
plt.ylabel("Complaint count")
plt.title("Complaint count by Median household income")
plt.show()

#bin by household income
#define bins & label. Use pd.cut to bin on school_sumry_df dataframe
bins = [0, 20000, 40000, 60000, 80000, 180000]
bin_labels = ["<20K","20K-40K","40K-60K","60K-80K", ">80K",]
zipcode_main_df["median_household_income_bins"] = pd.cut(zipcode_main_df["median_household_income($)"], bins, right = False, labels = bin_labels)

income_bar = pd.DataFrame(zipcode_main_df.groupby("median_household_income_bins")["complaint_count"].mean())
income_bar = income_bar.reset_index()
income_bar

# Create the ticks for our bar chart's x axis
#plt.figure(figsize=(20,3))
x_axis = np.arange(len(income_bar["median_household_income_bins"]))
tick_locations = [value + 0.4 for value in x_axis]
plt.xticks(tick_locations, income_bar["median_household_income_bins"])

plt.bar(x_axis, income_bar["complaint_count"], alpha=0.5, align="edge",color = "green", edgecolor = "r")

plt.xlabel("Median household income")
plt.ylabel("Complaint Count")
plt.title("Complaint count by Median household income")

plt.show()

#complaint count vs population below poverty line
plt.scatter (zipcode_main_df["population_below_poverty_level(%)"], zipcode_main_df["complaint_count"])
plt.xlabel("Population below poverty level(%)")
plt.ylabel("Complaint count")
plt.title("Complaint count by population below poverty level")
plt.show()

#complaint count vs crime rate
plt.scatter (zipcode_main_df["crime_reports"], zipcode_main_df["complaint_count"])
plt.xlabel("Crime rate")
plt.ylabel("Complaint count")
plt.title("Complaint count vs Crime rate")
plt.show()

#Regression analysis(best fit)
crime = zipcode_main_df.loc[:,"crime_reports"]
complaint = zipcode_main_df.loc[:,"complaint_count"]

#use mask to remove NaN
mask = ~np.isnan(crime) & ~np.isnan(complaint)
(slope, intercept, _, _, _) = linregress(crime[mask], complaint[mask])
fit = slope * crime[mask] + intercept

#plot crime reports vs complaint count
plt.scatter(crime[mask], complaint[mask], color='red', edgecolor='r')
plt.plot(crime[mask], fit, color='blue', linestyle='dashed')
plt.xlabel("Crime rate")
plt.ylabel("Complaint count")
plt.title("Complaint count vs Crime rate (regression line)")
plt.show()

#Regression analysis(best fit) when only looking at zip-code with population 20K-30K
crime = zipcode_20k_to_30k.loc[:,"crime_reports"]
complaint = zipcode_20k_to_30k.loc[:,"complaint_count"]

#use mask to remove NaN
mask = ~np.isnan(crime) & ~np.isnan(complaint)
(slope, intercept, _, _, _) = linregress(crime[mask], complaint[mask])
fit = slope * crime[mask] + intercept

#plot crime reports vs complaint count for zip-codes with population 20K-30K
plt.scatter(crime[mask], complaint[mask], color='red', edgecolor='r')
plt.plot(crime[mask], fit, color='blue', linestyle='dashed')
plt.xlabel("Crime rate")
plt.ylabel("Complaint count")
plt.title("Complaint count vs Crime rate for population between 20K-30K (regression line)")
plt.show()

#bin by renters & home-owners
#define bins & label. Use pd.cut to bin on school_sumry_df dataframe
bins = [0, 50, 100]
bin_labels = ["Home owners","Renters"]
zipcode_main_df["Majority"] = pd.cut(zipcode_main_df["renters(%)"], bins, right = False, labels = bin_labels)
zipcode_main_df.head()

majority_bar = pd.DataFrame(zipcode_main_df.groupby("Majority")["complaint_count"].mean())
majority_bar = majority_bar.reset_index()
majority_bar

# Create the ticks for our bar chart's x axis

x_axis = np.arange(len(majority_bar["Majority"]))
tick_locations = [value + 0.4 for value in x_axis]
plt.xticks(tick_locations, majority_bar["Majority"])

plt.bar(x_axis, majority_bar["complaint_count"], alpha=0.5, align="edge", color = "green", edgecolor = "r")

plt.xlabel("Majority in zipcode")
plt.ylabel("Complaint Count")
plt.title("Complaint count by Renters or Home-owner majority")

plt.show()



