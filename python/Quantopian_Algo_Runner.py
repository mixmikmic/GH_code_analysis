from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
import time
import datetime as dt
import re
import numpy as np
import pandas as pd
from stock_scraper import get_stock_prices
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d
import os

import requests
from bs4 import BeautifulSoup

from Markowitz import get_vol, opt_weight

get_ipython().run_line_magic('matplotlib', 'inline')

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--no-sandbox')

from pyvirtualdisplay import Display
display = Display(visible=0, size=(1024, 768)) 
display.start()

plt.rcParams['figure.figsize'] = (15,10)

from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')

def convert_to_df(df_str):
    header = re.findall("[ , \w]+\\n", df_str)[1]
    columns = re.findall("[\w]+", header)
    number_cols = len(columns) + 1 # Including index as column
    
    n = len(columns)
    all_numbers = re.findall("[-,.,0-9]+", df_str)
    values = [float(all_numbers[i]) for i in range(len(all_numbers)) if i % number_cols != 0]
    vals_array = np.array(values).reshape((int(len(values) / n), n))
    
    tickers = re.findall('\[(.*)\]', df_str)
    
    result_df = pd.DataFrame(index=tickers, data=vals_array, columns=columns)
    
    return result_df

driver = webdriver.Chrome("{}/chromedriver".format(os.getcwd()), 
                          chrome_options=chrome_options,
                         service_args=['--verbose', '--log-path=/tmp/chromedriver.log'])

# Get to algorithm login page
driver.get("https://www.quantopian.com/live_algorithms/5ac7078347a9990019bd2395")

# Log in and proceed to live trading page
username = driver.find_element_by_id("user_email")
password = driver.find_element_by_id("user_password")
username.send_keys("cib.pairs@gmail.com")
password.send_keys("CIBPairs17")
password.send_keys(Keys.ENTER)
time.sleep(0.5)

# Go to the logs pane
driver.find_element_by_css_selector("i.fontello-icon.fontello-icon-book").click()
        
    
# Wait until logs are displayed
print("Waiting for logs...")
weights_showing = False
while not weights_showing:
    try:
        driver.find_element_by_css_selector("div.logs-footer.hidden")
    except NoSuchElementException:
        time.sleep(3)
        continue
        
    print("Done")
    weights_showing = True
    
    
# Extract weights and tickers
weights_msg = driver.find_elements_by_xpath("//*[@class='logmsg' and contains(text(), 'WEIGHTS:')]")
weights_txt = weights_msg[0].text

tickers = re.findall('\[(.*)\]', weights_txt)
weights = re.findall('[-,.,0-9]+\\n', weights_txt)
weights = [el.replace('\n', '') for el in weights]
weights = [float(weight) for weight in weights]
       
portfolio = pd.Series(index = tickers, data = weights)
print("\nAlgo picks:\n" + str(portfolio))

# Extract scores of stocks picked
scores_msg = driver.find_elements_by_xpath("//*[@class='logmsg' and contains(text(), 'FACTOR SCORES:')]")
scores_txt = scores_msg[0].text
scores_df = convert_to_df(scores_txt)

# Extract scores of bottom ranked stocks
bottom_scores_msg = driver.find_elements_by_xpath("//*[@class='logmsg' and contains(text(), 'BOTTOM SCORES:')]")
bottom_scores_txt = bottom_scores_msg[0].text
bottom_scores_df = convert_to_df(bottom_scores_txt)

# Extract scores of top ranked stocks
top_scores_msg = driver.find_elements_by_xpath("//*[@class='logmsg' and contains(text(), 'TOP SCORES:')]")
top_scores_txt = top_scores_msg[0].text
top_scores_df = convert_to_df(top_scores_txt)


print("\nPortfolio Scores:\n", scores_df)
print("\nBottom Scores:\n", bottom_scores_df)
print("\nTop Scores:\n", top_scores_df)

time.sleep(5)
driver.close()

SPY_EXPECTED_RETURN = 0.08
def get_expected_returns(tickers, scores, portfolio_expected_return=SPY_EXPECTED_RETURN):
    url = "http://finance.yahoo.com/quote/{ticker}/"
    mkt_caps = np.ones((len(tickers), 1))
    with requests.Session() as s:
        for i, ticker in enumerate(tickers):
            page = s.get(url.format(ticker=ticker))
            soup = BeautifulSoup(page.content, 'html.parser')
            mkt_cap = soup.findAll("td", {"data-test":"MARKET_CAP-value"})[0].text
            scale = mkt_cap[-1]
            value = float(mkt_cap[:-1])
            if scale == 'M':
                value /= 1000.0
            mkt_caps[i, 0] = value
    
    w = mkt_caps / sum(mkt_caps)
    s = np.array(scores).reshape((len(scores), 1))
    score_scaler = portfolio_expected_return / (w.T @ s)
    print("Score Scaler: ", score_scaler[0][0])
    score_scaler = np.abs(score_scaler[0][0]) # TODO: Potentially problematic. Need to address
    result = pd.Series(index=tickers, data=(score_scaler * scores))
            
    return result

returns = get_expected_returns(portfolio.index, scores_df['composite'])

print("Raw Portfolio Picks")
sns.heatmap(pd.DataFrame(portfolio/2, columns=['Weights']), annot=True, cmap='RdYlGn')

print("Portfolio Factor Analysis")
sns.heatmap(scores_df.drop(['composite'], axis=1), annot=True, cmap='RdYlGn')

COVAR_LOOKBACK = 90 # In days, not just business days
today = dt.date.today()
start = dt.date.today() - dt.timedelta(COVAR_LOOKBACK)
prices = get_stock_prices(portfolio.index, str(start), str(today))['Adj Close']

try:
    null_locs = np.where(prices == 'null')
    for i in range(len(null_locs[0])):
        prices.iloc[null_locs[0][i], null_locs[1][i]] = np.nan
except:
    pass
prices = prices.fillna(method='pad')
prices = prices.apply(pd.to_numeric)
prices = prices.dropna(thresh=int(len(prices)*0.75), axis=1)

rets = prices.pct_change()

A = rets.dropna().T
A -= np.mean(A, axis=0)

cum_returns = (1 + rets.fillna(0)).cumprod()
np.log(cum_returns).plot(title='%d-Day Algo Picks\' Cumulative Log Returns' % COVAR_LOOKBACK)

A

from sklearn.decomposition import PCA
def visualize_PCA(data, num_components=3, title='pca'):
    samples = data
    pca = PCA(n_components=num_components, svd_solver='full')
    x_new = pca.fit_transform(samples)
    if num_components == 2:
        x_new = np.array([x_new[:, 0], x_new[:, 1]])
        plt.scatter(x_new[0], x_new[1])
        plt.title(title)
        for i in range(len(data.index)):
            plt.annotate(data.index[i], xy=(x_new[0, i],x_new[1, i]), xytext=(15,0), textcoords='offset points')
        plt.show()
    elif num_components == 3:
        x_new = np.array([x_new[:, 0], x_new[:, 1], x_new[:, 2]])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_new[0], x_new[1], x_new[2])
        for i in range(len(A)): #plot each point + it's index as text above
            ax.text(x_new[0,i],x_new[1,i],x_new[2,i],  A.index[i], size=10, zorder=1,  
            color='k') 
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

visualize_PCA(A, num_components=2)

# Now let's project on 3 dimensions
visualize_PCA(A)

print("3D PRINCIPLE COMPONENTS PROJECTION OF PAST %d DAY RETURNS" % COVAR_LOOKBACK)

from sklearn.manifold import TSNE
def visualize_TSNE(data, title='tsne'):
    samples = data
    tsne = TSNE(n_components=2)
    x_new = tsne.fit_transform(samples)
    x_new = np.array([x_new[:, 0], x_new[:, 1]])
    plt.scatter(x_new[0], x_new[1])
    for i in range(len(data.index)):
            plt.annotate(data.index[i], xy=(x_new[0, i],x_new[1, i]), xytext=(15,0), textcoords='offset points')
    plt.title(title)
    plt.show()

print('2D TSNE of past 90 Day Returns')
visualize_TSNE(A)

cov_mat = rets.dropna().cov()
print("COVARIANCE MATRIX FROM PAST %d DAYS RETURNS" % COVAR_LOOKBACK)
sns.heatmap(cov_mat, annot=True, center=0)

from sklearn.cluster import KMeans
def corr_matrix_data(X, num_clusters=3, x_labels = False):
    # demean the data
    X_demeaned = X - np.mean(X, 0) 
    
    _, _, V = np.linalg.svd(X_demeaned)
    V = V.T
    
    # get projected data
    X_proj = X_demeaned @ V[:, :num_clusters]
    
    # do some k-means clustering to identify which points are in which cluster
    km = KMeans(n_clusters = num_clusters)
    clusters = km.fit_predict(X_proj)
    
    # sort data based on identified clusters 
    t = X.copy()
    t['cluster'] = clusters
    t = t.sort_values("cluster")
    t = t.drop("cluster", 1)
    plt.imshow(t.T.corr(), "hot")
    plt.colorbar()
    plt.grid(False)
    plt.yticks(range(len(X)), t.index)
    if x_labels:
        plt.xticks(range(len(X)), t.index)
    

print("Unclustered Correlation Matrix Heat Map")
plt.imshow(rets.dropna().corr(), "hot")
plt.colorbar()
plt.grid(False)
_ = plt.xticks(range(rets.shape[1]), rets.columns)
_ = plt.yticks(range(rets.shape[1]), rets.columns)

NUM_CLUSTERS = 2
print("%d-Clustered Correlation Matrix Heat Map" % NUM_CLUSTERS)
clusters = corr_matrix_data(rets.dropna().T, num_clusters=NUM_CLUSTERS)

print("Groupings:\n" + str(clusters))

return_sd = pd.DataFrame(index = A.index)
return_sd['E[r]'] = returns
return_sd['vol'] = [np.sqrt(cov_mat.iloc[i, i]) for i in range(len(cov_mat))]

def get_MVP(mu_vec, cov_mat, mark_vols, mark_rets, stocks):
    MVP_index = np.argmin(mark_vols)
    MVP = opt_weight(mark_rets[MVP_index], cov_mat, mu_vec)
    MVP_series = pd.Series(index=stocks, data=MVP.reshape((len(MVP))))
    print("Minimum Variance Portfolio:\n" + str(MVP_series.sort_values()))
    
    return MVP_series

def get_market_port(mu_vec, cov_mat, mark_vols, mark_rets, rf, stocks):
    derivatives = []
    for i in range(1, len(mark_rets) - 1):
        derivative = (mark_rets[i + 1] - mark_rets[i - 1]) / (mark_vols[i + 1] - mark_vols[i - 1])
        derivatives.append(derivative)
        
    cap_mkts_slopes = []
    for i in range(1, len(mark_rets) - 1):
        cap_mkts_slope = (mark_rets[i] - rf) / mark_vols[i]
        cap_mkts_slopes.append(cap_mkts_slope)
        
    MVP_index = np.argmax(derivatives)
    market_portfolio_index = np.argmin((np.array(derivatives[MVP_index:]) - np.array(cap_mkts_slopes[MVP_index:]))**2)
    market_portfolio_index += MVP_index
    market_portfolio_ret = mark_rets[market_portfolio_index - 1]
    market_portfolio = opt_weight(market_portfolio_ret, cov_mat, mu_vec)
    market_port_series = pd.Series(index=stocks, data=market_portfolio.reshape((len(market_portfolio))))
    
    cap_mkt_slope = cap_mkts_slopes[market_portfolio_index]
    
    print("Market Portfolio, assuming risk-free rate of %.2f:\n" % rf + str(market_port_series.sort_values()))
    
    return market_port_series, cap_mkt_slope
    

RF_RATE = 0.03

def plot_eff_front(mu_vec, cov_mat, stocks):
    mu_vec = mu_vec.reshape((len(mu_vec), 1))
    rets = np.linspace(-2 * np.abs(min(mu_vec)), 2*np.abs(max(mu_vec)), 100)
    vols = [get_vol(opt_weight(r, cov_mat, mu_vec), cov_mat) for r in rets]
    
    MVP = get_MVP(mu_vec, cov_mat, vols, rets, stocks)
    market_port, cap_mkt_slope = get_market_port(mu_vec, cov_mat, vols, rets, RF_RATE, stocks)
    x = np.linspace(0, max(vols), 100)
    cap_mkts_line = [RF_RATE + cap_mkt_slope*x for x in x]
    
    plt.plot(vols, rets)
    plt.plot(x, cap_mkts_line)
    plt.xlim(xmin=0)
    
    return MVP, market_port

print("Return-Volatility Table:\n", return_sd)

return_sd.plot.scatter(x='vol', y='E[r]')
MVP1, market_port1 = plot_eff_front(np.array(return_sd['E[r]']), np.array(cov_mat), return_sd.index)

def unlever_portfolio(portfolio):
    unlevered_port = portfolio.copy()
    short_sum = 0
    long_sum = 0
    for weight in unlevered_port:
        if weight < 0:
            short_sum -= weight
        else:
            long_sum += weight
            
    for i in range(len(unlevered_port)):
        if unlevered_port[i] < 0:
            unlevered_port[i] /= short_sum
        else:
            unlevered_port[i] /= long_sum

    if short_sum != 0:
        unlevered_port /= 2
            
    return unlevered_port

unlevered_MVP1 = unlever_portfolio(MVP1)
print("Unlevered Minimum Variance Portfolio")
sns.heatmap(pd.DataFrame(unlevered_MVP1, columns=['Weights']).sort_values('Weights'), annot=True, cmap='RdYlGn', center=0)

unlevered_market_port1 = unlever_portfolio(market_port1)
print("Unlevered Market Portfolio")
sns.heatmap(pd.DataFrame(unlevered_market_port1, columns=['Weights']).sort_values('Weights'), annot=True, cmap='RdYlGn', center=0)

all_scores = pd.concat([bottom_scores_df, top_scores_df], axis=0)
returns = get_expected_returns(all_scores.index, all_scores['composite'])

print("Factor Analysis")
sns.heatmap(all_scores.drop(['composite'], axis=1), annot=True, cmap='RdYlGn')

COVAR_LOOKBACK = 90 # In days, not just business days
today = dt.date.today()
start = dt.date.today() - dt.timedelta(COVAR_LOOKBACK)
prices = get_stock_prices(all_scores.index, str(start), str(today))['Adj Close']

try:
    null_locs = np.where(prices == 'null')
    for i in range(len(null_locs[0])):
        prices.iloc[null_locs[0][i], null_locs[1][i]] = np.nan
except:
    pass
prices = prices.fillna(method='pad')
prices = prices.apply(pd.to_numeric)
prices = prices.dropna(thresh=int(len(prices)*0.75), axis=1)

rets = prices.pct_change()

A = rets.dropna().T
A -= np.mean(A, axis=0)


cum_returns = (1 + rets.fillna(0)).cumprod()
top_movers = np.abs(cum_returns.iloc[-1] - 1).sort_values(ascending=False).head(10).index
np.log(cum_returns)[top_movers].plot(title='%d-Day Top Movers\' Cumulative Log Returns' % COVAR_LOOKBACK)

visualize_PCA(A, num_components=2)

# Now let's project on 3 dimensions
visualize_PCA(A)

print('2D TSNE of past 90 Day Returns')
visualize_TSNE(A)

cov_mat = rets.dropna().cov()
print("COVARIANCE MATRIX FROM PAST %d DAYS RETURNS" % COVAR_LOOKBACK)
sns.heatmap(cov_mat, annot=False, center=0)

print("Unclustered Correlation Matrix Heat Map")
plt.imshow(rets.dropna().corr(), "hot")
plt.colorbar()
plt.grid(False)
_ = plt.yticks(range(rets.shape[1]), rets.columns)

NUM_CLUSTERS = 3
print("%d-Clustered Correlation Matrix Heat Map" % NUM_CLUSTERS)
clusters = corr_matrix_data(rets.dropna().T, num_clusters=NUM_CLUSTERS)

print("Groupings:\n" + str(clusters))

return_sd = pd.DataFrame(index = A.index)
return_sd['E[r]'] = returns
return_sd['vol'] = [np.sqrt(cov_mat.iloc[i, i]) for i in range(len(cov_mat))]

print("Using scores as expected returns, Return-Volatility Table:\n", return_sd)

return_sd.plot.scatter(x='vol', y='E[r]')
MVP2, market_port2 = plot_eff_front(np.array(return_sd['E[r]']), np.array(cov_mat), return_sd.index)

unlevered_MVP2 = unlever_portfolio(MVP2)
print("Unlevered Minimum Variance Portfolio")
sns.heatmap(pd.DataFrame(unlevered_MVP2, columns=['Weights']).sort_values('Weights'), annot=True, cmap='RdYlGn', center=0)

unlevered_market_port2 = unlever_portfolio(market_port2)
print("Unlevered Market Portfolio")
sns.heatmap(pd.DataFrame(unlevered_market_port2, columns=['Weights']).sort_values('Weights'), annot=True, cmap='RdYlGn', center=0)

