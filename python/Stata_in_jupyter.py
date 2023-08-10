get_ipython().magic('matplotlib inline')
import seaborn as sns
import pandas as pd
import statsmodels.formula.api as smf
import ipystata

get_ipython().run_cell_magic('stata', '-o life_df', 'sysuse lifeexp.dta\nsummarize')

life_df.head(3)

get_ipython().run_cell_magic('stata', '-o life_df', 'gen lngnppc = ln(gnppc)\nregress lexp lngnppc')

model = smf.ols(formula = 'lexp ~ lngnppc',
                data = life_df)
results = model.fit()

print(results.summary())

life_df.popgrowth = life_df.popgrowth * 100

life_df.popgrowth.mean()

get_ipython().run_cell_magic('stata', '-d life_df', 'summarize')

get_ipython().run_cell_magic('stata', '-d life_df --graph ', 'graph twoway (scatter lexp lngnppc) (lfit lexp lngnppc)')

sns.set_style("whitegrid")
g=sns.lmplot(y='lexp', x='lngnppc', col='region', data=life_df,col_wrap=2)

