# Import libraries
import pandas as pd

# Read the data
def read_data():
    df = pd.read_csv("data/AAPL.csv")
    print df.tail()

if __name__ == "__main__":
    read_data()

