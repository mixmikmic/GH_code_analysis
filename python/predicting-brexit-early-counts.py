import random

import numpy as np
import scipy as sp
import pandas as pd

import scipy.stats

import pystan as ps

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
# import seaborn as sns; sns.set_context('notebook')

import toyplot as tp
import toyplot.svg

df = pd.read_csv("brexit-data/data654pm.csv")

print(len(df))

df["total"] = df["leave"] + df["remain"]
df["frac_leave"] = df["leave"] / df["total"]

# remove gibraltar outlier
df = df[~df.area.str.contains("gibraltar")]

df.head(10)

canvas = tp.Canvas(600, 350)
axes = canvas.cartesian(label="Rseults of early votes by district", xlabel="# of votes (x1000)", ylabel="% leave")
axes.scatterplot(df["total"]/1000, df["leave"]/df["total"])
axes.hlines([0.5], color="grey")
axes.x.ticks.show = True
axes.y.ticks.show = True
axes.y.ticks.locator = tp.locator.Explicit(locations=np.linspace(0, 1, 11))

# svg = toyplot.svg.render(canvas, "plots/early_results_scatter.svg")

canvas = tp.Canvas(600, 350)
axes = canvas.cartesian(xlabel="Count", ylabel="% Leave")
axes.bars(np.histogram(df["leave"] / df["total"], bins=10))

total_prob = df["leave"].sum() / df["total"].sum()
print("{:.2f}% votes to LEAVE out of counted votes".format(100*total_prob))
print("total votes: {} - {} \t total: {} from {} districts".format(df.leave.sum(), 
                                                                   df.remain.sum(), 
                                                                   df.total.sum(),
                                                                   len(df)))

(df.leave.sum() - 0.5 * df.total.sum()) / np.sqrt(2 * df.total.sum())

simple_model = """
data {
    int K;
    int total[K];
    int vote_leave[K];
}
parameters {
    real<lower=1> alpha;
    real<lower=1> beta;
    real<lower=0, upper=1> p[K];
}
model {
    for (i in 1:K) {
        p[i] ~ beta(alpha, beta);
        vote_leave[i] ~ binomial(total[i], p[i]);
    }
}
"""

data = {'K': len(df),
               'total': df["total"],
               'vote_leave': df["leave"]}

fit = ps.stan(model_code=simple_model, 
                         data=data, iter=5000, chains=4)

fit.plot(pars=['alpha', 'beta']);

overall_outcomes = fit["alpha"] / (fit["alpha"] + fit["beta"])

outcomes = []
for alpha, beta in zip(fit["alpha"], fit["beta"]):
    # new outcomes
    populations = np.random.choice(df["total"], size=len(df))
    p_leaves = [random.betavariate(alpha, beta) for _ in populations]
    leave_votes = [np.random.binomial(pop, prob) for pop, prob in zip(populations, p_leaves)]

    total_votes = sum(populations)
    total_leave = sum(leave_votes)
    outcomes.append(total_leave / total_votes)

canvas = tp.Canvas(600, 350)
axes = canvas.cartesian(label="Posterior for P(leave)", xlabel="% votes to leave", ylabel="density")
axes.bars(np.histogram(outcomes, bins=100, normed=True))
axes.vlines([0.5])

axes.x.ticks.show = True
axes.y.ticks.show = True

# svg = toyplot.svg.render(canvas, "plots/monte_carlo_results.svg")

canvas = tp.Canvas(600, 350)
axes = canvas.cartesian(label="Density of posterior for leave", xlabel="% votes to leave", ylabel="density")
axes.vlines([0.5], color="grey")

axes.scatterplot(df.frac_leave, [0.05 for _ in range(len(df))], marker="|", color="black", size=10)

x = np.linspace(0, 1, 100)

for alpha, beta in zip(fit["alpha"], fit["beta"]):
    if random.random() < 0.02:
        axes.plot(x, sp.stats.beta.pdf(x, alpha, beta), color="steelblue", opacity=0.1)

prob_leave_wins = len(list(filter(lambda x: x>0.5, outcomes)))/len(outcomes)
print("Probability LEAVE wins: {:.2f}%".format(100*prob_leave_wins))

