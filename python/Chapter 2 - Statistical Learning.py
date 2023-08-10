import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

advert = pd.read_csv('../../data/Advertising.csv', index_col=0)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

sns.regplot(x='TV', y='sales', data=advert, ax=ax1, scatter_kws={'color': 'red'}, 
            line_kws={'color': 'blue'}, ci=None)

sns.regplot(x='radio', y='sales', data=advert, ax=ax2, scatter_kws={'color': 'red'}, 
            line_kws={'color': 'blue'}, ci=None)

sns.regplot(x='newspaper', y='sales', data=advert, ax=ax3, scatter_kws={'color': 'red'}, 
            line_kws={'color': 'blue'}, ci=None)



