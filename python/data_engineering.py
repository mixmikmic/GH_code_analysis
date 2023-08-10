# Dependencies
import numpy as np
import pandas as pd

# Read weather data
df_measurements = pd.read_csv("Resources/hawaii_measurements.csv")
df_stations = pd.read_csv("Resources/hawaii_stations.csv")

# Determine count of dataset
print(f"Length of measurements dataset: {len(df_measurements)}")

# Visualize measurements dataframe
df_measurements.head(50)

# Set the index name so it's called "id" when we have it in the cleaned CSV
df_measurements.index.name = 'id'

# Clean the NaN's from the precipitation column (convert to zero)
df_measurements['prcp'] = df_measurements['prcp'].fillna(0)

df_measurements.head(100)

# Write cleaned measurements dataframe to fresh file
df_measurements.to_csv("Resources/clean_hawaii_measurements.csv", encoding="utf-8", index=True)

# Visualize stations dataframe (no need to clean it up, since there are no NaN values)
df_stations.head(100)



