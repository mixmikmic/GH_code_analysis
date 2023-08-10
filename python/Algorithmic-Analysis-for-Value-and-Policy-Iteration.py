get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def series(discount_factor, n=1e6):
    assert 0 < discount_factor < 1, "Discount factor must be >0 and <1"
    return sum([discount_factor ** i for i in range(int(n))])

betas = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

# By calculation
plt.plot(betas, [series(beta) for beta in betas]);

# By formula (beta / (1 - beta))
plt.plot(betas, [beta / (1 - beta) for beta in betas]);

def action1(beta, t):
    return sum([-beta ** t for t in range(t)])

def action2(beta, t):
    return -beta ** 2 / (1 - beta)

betas = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
steps = 20

rows = []
for beta in betas:
    for step in range(steps):
        rows.append({
            'step': step,
            'beta': beta,
            'a1': action1(beta, step),
            'a2': action2(beta, step)
        })
        
df = pd.DataFrame(rows)
df_ = pd.melt(df, id_vars=['step', 'beta'])

g = sns.FacetGrid(df_, col='beta', hue='variable', sharey=False)
g.map(sns.pointplot, 'step', 'value');

