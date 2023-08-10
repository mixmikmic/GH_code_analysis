get_ipython().magic('pylab inline')

import pandas as pd
import numpy as np

# Seaborn requires matplotlib package to be installed
# https://stanford.edu/~mwaskom/software/seaborn/installing.html
#
# If it is not available, use:
# pip install matplotlib
# 
# If you are using Anaconda distribution, then
# conda install matplotlib
import seaborn as sb
import json
import datetime

# Read and inspect the data file
df = pd.read_csv("product_test_data.csv")
print df.head(10)
print df.columns
print df.dtypes

# Convert the string which has a list of values to an actual python list
df["Amount"] = df["Amount"].apply(json.loads)

# Create a new column which has the sum of production application
df["Total_Amount"] = df["Amount"].apply(sum)

# Create a new column for the number of entries
df["No_of_entries"] = df["Amount"].apply(len)

# Remove unused columns
df.drop(["Amount"], axis=1, inplace=True)

# Check if the transformations have been successful
print df.head(10)
print df.columns

# Perform a Group-By operation on "User_ID" and sum up the "Total_Amount" field
top10_users_products = df.groupby(['User_ID'], as_index=False)['Total_Amount'].sum()

# Sort in descending order based on "Total_Amount" field
top10_users_products.sort_values("Total_Amount", ascending=False, inplace=True)

# By default, pandas retains the index values as in the original dataframe.
# Reset the index to start from beginning
top10_users_products.reset_index(inplace=True)

# Show only the top 10 records
print top10_users_products.head(10)

# Filter rows for "Product1"
top3_product1 = df[df["Product"] == "Product1"]

# Extract the columns - "User_ID" and "No_of_entries"
top3_product1 = top3_product1[["User_ID", "No_of_entries"]]

# Sort on "No_of_entries" column in descending order
top3_product1.sort_values("No_of_entries", ascending=False, inplace=True)

# Display top 3 rows
print top3_product1.head(3)

product_max_usage = df.groupby(["Product"], as_index=False)["Total_Amount"].sum()
print product_max_usage.max(column="Total_Amount")

# Survey duration is given as 90 days
SURVEY_DURATION = 90

# Take the current time
today_date = datetime.datetime.today()

# Calculate the start date as 90 days prior
start_date = today_date - datetime.timedelta(days=SURVEY_DURATION)

# Convert the date that is a string to YY-MM-DD format and
# find the number of days elapsed from the start date
df["Days"] = (df["Entry_Date"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d")) - start_date) / np.timedelta64(1, "D")

# Round off the day values to a whole number
df["Days"] = df["Days"].round(0)

# Calculate the week by dividing the number of days by 7.
# Add 1 to start the week count from 1 instead of 0
df["Week"] = ((df["Days"] / 7) + 1).round()

# Remove the "Days" column
df.drop(["Days"], axis=1, inplace=True)


# Group by "Product" and "Week" fields followed by summation over "Total_Amount" field
weekly_usage_all_products = df.groupby(["Product", "Week"], as_index=False)["Total_Amount"].sum()

# Sort by "Week" and then "Product"
weekly_usage_all_products.sort_values(["Week", "Product"], inplace=True)

print weekly_usage_all_products

# Plotting the above data using seaborn package
print sb.factorplot(
    x="Week", y="Total_Amount",
    hue="Product",
    data=weekly_usage_all_products,
    size=12,
    kind="bar",
    palette="muted"
)



