import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import plotnine as plt
from plotnine import *

# Read in the data and set the proper timezone
sales = pd.read_csv('https://byuistats.github.io/M335/data/sales.csv',
                    parse_dates=['Time'])
sales['Time'] = (sales['Time'].dt.tz_localize('UTC')
                              .dt.tz_convert('US/Mountain')
                              .dt.tz_localize(None))

# Discard outlying and 'missing' transactions
sales = sales[(sales['Amount'] < 500) & (sales['Name'] != 'Missing')]

# Aggregate the sales amounts by hour
sales['Time'] = sales['Time'].dt.floor('60min')
sales_hourly = pd.DataFrame(sales.groupby(['Name','Time'])['Amount'].sum())
sales_hourly = sales_hourly.reset_index()

# Add a column for the hour and discard the two transactions that took place
# before 5:00 AM
sales_hourly['Hour'] = sales_hourly['Time'].dt.hour
sales_hourly = sales_hourly[sales_hourly['Hour'] > 5]

# Plot the number of transaction vs. time of day
sales_count = (pd.DataFrame(sales_hourly.groupby(['Name', 'Hour'])['Amount']
                 .count())
                 .reset_index())
(ggplot(sales_count, aes(x='Hour', y='Amount')) +
 geom_col() +
 facet_grid('. ~ Name') +
 labs(title='Number of Transactions vs Time of Day',
      y='Number of Transactions') +
 theme_bw())

# List for ordering days of week
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

# Add column for day of week
sales_hourly['Day'] = sales_hourly['Time'].dt.weekday_name
sales_daily_count = (pd.DataFrame(sales_hourly.groupby(['Name','Day'])['Amount']
                       .count())
                       .reset_index())

# Plot sales vs day of week
plt.options.figure_size = (17, 4.8)
(ggplot(sales_daily_count[sales_daily_count['Day'] != 'Sunday'], aes(x='Day', y='Amount')) +
 geom_col() +
 facet_grid('. ~ Name') +
 scale_x_discrete(limits=weekday_order) +
 labs(title='Number of Transaction by Day of Week',
      x='Weekday',
      y='Number of Transactions') +
 theme_bw() +
 theme(axis_text_x = element_text(angle = 55, vjust = 1, hjust = 1)))

# Plot the sales amount vs. time of day
plt.options.figure_size = (18, 4.8)
(ggplot(sales_hourly, aes(x='Hour', y='Amount')) +
 geom_point() +
 geom_smooth(color='blue') +
 facet_grid('. ~ Name') +
 labs(title='Transaction Amount vs Time of Day') +
 theme_bw())

# Plot the companies vs. transaction amount
plt.options.figure_size = (10, 4.8)
(ggplot(sales_hourly, aes(x='Name', y='Amount')) +
 geom_boxplot() +
 labs(title='Average Transaction Amount by Company',
      x='Company') +
 theme_bw())

# A list of the companies in order of profit
ordered_by_profit = ['HotDiggity','LeBelle','Tacontento','SplashandDash','ShortStop','Frozone']

# Plot the sum of all transations for each company
sales_total = pd.DataFrame(sales_hourly.groupby(['Name'])['Amount'].sum()).reset_index()
(ggplot(sales_total, aes(x='Name', y='Amount')) +
 geom_col() +
 scale_x_discrete(limits=ordered_by_profit) +
 labs(title='Total Semester Revenue by Company',
      x='Company') +
 theme_bw())

