import pandas as pd
subway_df = pd.read_csv('nyc_subway_weather.csv')

subway_df.head()

subway_df.describe().transpose()

# Create function to calculate Pearson's R
def correlation(x,y):
    std_x = (x - x.mean())/x.std(ddof = 0)
    std_y = (y - y.mean())/y.std(ddof = 0)
    
    return (std_x * std_y).mean()

correlation(subway_df['ENTRIESn_hourly'], subway_df['meanprecipi'])

correlation(subway_df['ENTRIESn_hourly'],subway_df['ENTRIESn'])

