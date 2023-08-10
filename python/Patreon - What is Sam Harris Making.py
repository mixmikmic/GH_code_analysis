import qgrid
import numpy as np
import pymc3 as pm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as graph
from IPython.display import display, Markdown
from tqdm import tqdm
from selenium import webdriver

graph.style.use('fivethirtyeight')

# Open Chrome
broswer = webdriver.Chrome()

# Go to website
broswer.get('https://graphtreon.com/all-patreon-creators')
# broswer.implicitly_wait(10)  # Units are seconds

# Create Empty Dataframe
scraped = []

rows = broswer.find_elements_by_css_selector('tr')

print('N Rows:', len(rows))

for row in tqdm(rows, desc='Scraping'):
    cols = row.find_elements_by_css_selector('td')
    scraped.append([col.text for col in cols])

scraped = scraped[1:]
df = pd.DataFrame(scraped)
df.head()

# Clean Data Functions
def find_months(thing):
    if thing is None:
        return pd.np.nan
    elif 'month' in thing:
        return pd.np.nan
    else:
        return 1
    
# Clean Data
df['contribs'] = df[2].apply(lambda r: pd.np.NaN if r is None else r.replace(',', ''))
df['contribs'] = df['contribs'].astype(float)
df['dropme'] = df[3].apply(lambda x: find_months(x))
df['ppv'] = df[3].apply(lambda x: pd.np.nan if x is None else x.split(' ')[0])
df['ppv'] = df['ppv'].apply(lambda x: pd.np.nan if x is None or type(x) is float else x.replace(',', ''))
df['ppv'] = df['ppv'].apply(lambda x: pd.np.nan if x is None or type(x) is float else x.replace('$', ''))
df['ppv'] = df['ppv'].apply(lambda x: pd.np.nan if x == '' else x)
df['ppv'] = df['ppv'].astype(float)

data = df.dropna().drop('dropme', axis='columns')
print('N =', len(data))
display(data.head())

data.to_csv('data/filteredTop1000Patreons.csv')

data = pd.read_csv('data/filteredTop1000Patreons.csv')

graph.plot(data['contribs'], data['ppv'], 'o', alpha=0.5)
graph.show()

with pm.Model() as model:
    pm.glm.glm('ppv ~ contribs', data, family=pm.glm.families.StudentT())
    trace = pm.sample(3000, pm.NUTS(scaling=pm.find_MAP()), progressbar=True)

pm.traceplot(trace)
graph.show()

pm.plot_posterior(trace[1000:])
graph.show()

graph.plot(data['contribs'], data['ppv'], 'o', alpha=0.66)
graph.plot(data['contribs'], data['contribs'] * trace['contribs'][1000:].mean() + trace['Intercept'][1000:].mean())
graph.xlabel('Contributors')
graph.ylabel('Payment Per Post')
graph.show()

# Dealing with heteroskedasticity
with pm.Model() as hetero_model:
    # Priors
    m = pm.Normal('m', mu=0, sd=20)
    b = pm.Normal('b', mu=0, sd=500)
    noise = pm.HalfNormal('noise', sd=500)
    
    # Likelihood
    obs = pm.Cauchy('y', alpha=m * data['contribs'] + b, beta=noise, observed=data['ppv'])
    
    # Infer
    trace_cauchy = pm.sample(50000, step=pm.NUTS(), start=pm.find_MAP(), progressbar=True)

pm.traceplot(trace_cauchy)
graph.show()

pm.plot_posterior(trace_cauchy[1000::2])
graph.show()

final_trace = trace_cauchy[1000::2]

graph.plot(data['contribs'], data['ppv'], 'o', alpha=0.66)
graph.plot(data['contribs'], final_trace['m'].mean() * data['contribs'] + final_trace['b'].mean())
graph.xlabel('Contributors')
graph.ylabel('Payment Per Post')
graph.show()

df_trace = pm.trace_to_dataframe(final_trace)
display(df_trace.head())

# Query point for Sam Harris is 8416 Patreon contributors
df_trace['sam_hat'] = df_trace['m'] * 8416 + df_trace['b'] + df_trace['noise'].apply(
    lambda x: np.random.normal(0, x, size=1)[0]
)
display(df_trace.head())

graph.hist(df_trace['sam_hat'], bins=50)
graph.xlabel('Estimated Earnings Per Podcast ($)')
graph.show()

display(df_trace.describe())

credible_region = df_trace['sam_hat'].quantile(q=[0.05, 0.95]).values
print(credible_region)

display(Markdown('### 95% Chance Sam Harris makes \${:.2f} to \${:.2f} per podcast'.format(
    credible_region[0], 
    credible_region[1]
)))

