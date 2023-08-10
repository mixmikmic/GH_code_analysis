import pandas as pd
import re
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import *
init_notebook_mode(connected=True)

reverb_df = pd.read_csv('reverb_effects_pedals_v4_08032016.csv')
reverb_df.shape

reverb_df.head(3)

reverb_df.columns

reverb_df['id'].value_counts()[0:3]

reverb_df[reverb_df.id == 1363704].web_url

reverb_df = reverb_df.drop_duplicates('web_url', keep='first')

reverb_df.shape

reverb_df.make.unique()

def fix_make_names(row):
    r = row
    if 'Ibanez' in r:
        return 'Ibanez'
    elif 'WAMPLER' in r or 'wampler' in r:
        return 'Wampler'
    elif 'Mooer' in r:
        return 'Mooer'
    elif 'Dunlop' in r:
        return 'Dunlop'
    elif 'MXR' in r:
        return 'MXR'
    elif 'JHS' in r:
        return 'JHS'
    elif 'ZVEX' in r:
        return 'Zvex'
    return r

reverb_df.make = reverb_df.make.apply(fix_make_names)

reverb_df.make.value_counts()

reverb_df = reverb_df.groupby('make').filter(lambda x: len(x) > 100)
reverb_df.shape

def plot_histogram(x, xaxis_title, yaxis_title, chart_title, xaxis_tickfont_size=12, yaxis_tickfont_size=12):
    data = [Histogram(x=x)]
    layout = Layout(
        xaxis = dict(title=xaxis_title, tickfont=dict(size=xaxis_tickfont_size)),
        yaxis = dict(title=yaxis_title, tickfont=dict(size=yaxis_tickfont_size)),
        title = chart_title
    )
    fig = Figure(data=data, layout=layout)
    return fig

iplot(plot_histogram(reverb_df.make, None, 'Number of Pedals', 'Number of Pedals by Brand'))

reverb_df.condition.unique()

iplot(plot_histogram(reverb_df.condition, 'Condition Type', 'Number of Pedals', 
                     'Number of Pedals by Condition Type'))

reverb_df[reverb_df.condition == 'Non Functioning'].title

shop_name_counts = pd.DataFrame(reverb_df.shop_name.value_counts()).reset_index().sort_values(by='shop_name')
shop_name_counts.columns = ['shop_name', 'shop_count']
shop_name_counts.head(3)

def count_shops(shop_counts):
    shop_vals = list(shop_counts)
    count = 0
    for x in shop_vals:
        if x == 1:
            count += 1
    print(round(count / len(shop_vals), 2))
count_shops(reverb_df.shop_name.value_counts())

def plot_bar(x, y, xaxis_title, yaxis_title, chart_title, xaxis_tickfont_size=12, yaxis_tickfont_size=12):
    
    data = [Bar(x=x, y=y)]
    layout = Layout(
        xaxis = dict(title=xaxis_title, tickfont=dict(size=xaxis_tickfont_size)),
        yaxis = dict(title=yaxis_title, tickfont=dict(size=yaxis_tickfont_size)),
        title = chart_title
    )
    fig = Figure(data=data, layout=layout)
    return fig

iplot(plot_bar(shop_name_counts.shop_name, shop_name_counts.shop_count, None, 'Number of Pedals', 
               'Number of Pedals by Shop', 6))

reverb_df.shop_name.value_counts()[:3]

top_shops = shop_name_counts.sort_values(by='shop_count').tail(25)

iplot(plot_bar(top_shops.shop_name, top_shops.shop_count, None, 'Number of Pedals', 
               '# of Pedals Sold by Shop (Top 25 Stores)', 8))

reverb_df.year.unique()

def fix_years(row):
    if re.search('^(19|20)\d{2}$', str(row)):
        return row
    return None

reverb_df.year = reverb_df.year.apply(fix_years)
reverb_df.year.unique()

iplot(plot_histogram(reverb_df.year, 'Year', 'Number of Pedals', 'Number of Pedals by Year of Origin'))

iplot(plot_histogram(reverb_df.listing_currency, 'Currency', 'Number of Pedals', 'Number of Pedals by Currency'))

us_currency = reverb_df[reverb_df.listing_currency == 'USD']

us_currency['price.amount'].describe()

iplot(plot_histogram(us_currency['price.amount'], 'Price ($)', '# of Pedals', '# of Pedals by Price ($)'))

price_no_outliers = us_currency[us_currency['price.amount'] < 200]

price_no_outliers['price.amount'].describe()

iplot(plot_histogram(price_no_outliers['price.amount'], 'Price ($)', 'Number of Pedals', 
                     'Number of Pedals by Price ($)'))

reverb_df.ix[reverb_df['price.amount'].idxmax()]

reverb_df.ix[reverb_df['price.amount'].idxmin()]

round(1 - (reverb_df[reverb_df['price_drop.percent'].isnull()].shape[0] / reverb_df.shape[0]), 2)

iplot(plot_histogram(reverb_df['price_drop.percent'], 'Price Drop (%)', 'Number of Pedals', 
                     'Number of Pedals by % Price Drop'))

us_currency['shipping.us_rate.amount'].describe()

us_shipping_amount_clean = us_currency[(us_currency['shipping.us_rate.amount'] > 0) &                                        (us_currency['shipping.us_rate.amount'] <= 60)]

us_shipping_amount_clean['shipping.us_rate.amount'].describe()

iplot(plot_histogram(us_shipping_amount_clean['shipping.us_rate.amount'], 
                     'Shipping Cost ($)', '# of Pedals', 
                     '# of Pedals by Shipping Cost ($)'))

us_currency.ix[us_currency['shipping.us_rate.amount'].idxmax()]

