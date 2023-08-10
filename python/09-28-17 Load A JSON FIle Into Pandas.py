# Load library
import pandas as pd

# Create URL to JSON file (alternatively this can be a filepath)
url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/data.json'

# Load the first sheet of the JSON file into a dataframe
df = pd.read_json(url, orient='columns')

# View the first ten rows
df.head(10)

