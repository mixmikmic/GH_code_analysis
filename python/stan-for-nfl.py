import re

import pandas as pd
import numpy as np

import toyplot as tp
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import pystan as ps

raw_df = pd.read_csv("data/2016-week13.csv").drop("NA", 1)
raw_df.head()

team_to_id = { team: idx for idx, team in enumerate(raw_df["Loser"].factorize()[1]) }
id_to_team = { idx: team for idx, team in enumerate(raw_df["Loser"].factorize()[1]) }

def team_id(name):
    """ Function to find the team id corresponding to a team name"""
    try:
        return [idx for team, idx in team_to_id.items() if name.lower() in team.lower()][0]
    except:
        return None
    
def find_games(data, team):
    return data.loc[(traingames["tid"]==team_id(team)) | (traingames["oid"]==team_id(team))]

raw_df["win_id"] = raw_df.apply(lambda x: team_to_id[x.Winner], 1)
raw_df["los_id"] = raw_df.apply(lambda x: team_to_id[x.Loser], 1)
raw_df["at_home"] = raw_df.apply(lambda x: 0 if x.Location == "@" else 1, 1)

df = raw_df[raw_df["Week"] <= 14]
df.tail()

dict_df = df.to_dict(orient="index")

def extract_team_data(x):
    game_w = {"tid": x["win_id"], 
              "oid": x["los_id"],
              "win": 1,
              "at_home": x["at_home"],
              "week": x["Week"],
              "yds": x["YdsW"],
              "opp_yds" : x["YdsL"],
              "tow": x["TOW"],
              "tol": x["TOL"],
              "ptsf": x["PtsW"],
              "ptsa": x["PtsL"],
              "team": x["Winner"],
              "opp": x["Loser"]}
    game_l = {"tid": x["los_id"], 
              "oid": x["win_id"],
              "win": 0,
              "at_home": 1-x["at_home"],
              "week": x["Week"],
              "yds": x["YdsL"],
              "opp_yds" : x["YdsW"],
              "tow": x["TOL"],
              "tol": x["TOW"],
              "ptsf": x["PtsL"],
              "ptsa": x["PtsW"],
              "team": x["Loser"],
              "opp": x["Winner"]
            }
    if game_w["at_home"] == 1:
        return  game_w
    return game_l

allgames = pd.DataFrame([extract_team_data(x) for x in dict_df.values()])
allgames.head()

traingames = allgames[allgames["week"] < 14]
testgames = allgames[(allgames["week"] >= 14)]
testgames.head()

find_games(traingames, "denver")

stan_simple = """
data {
    int<lower=0> nteams; // number of teams
    int<lower=0> N; // number of observations
    int win[N]; // Outcome
    int tid[N]; // Team
    int oid[N]; // Opponent
    
    int Nnew; // new observations
    int tidnew[Nnew];
    int oidnew[Nnew];
}
transformed data {}
parameters {
    real home;
    real team[nteams];
    
    real<lower=0, upper=3> sigma_team;
    real<lower=0, upper=1> sigma_home;
}
transformed parameters {
    vector[N] xb;
    vector[N] pwin;
    for(i in 1:N) {
        xb[i] <- home + team[tid[i]] - team[oid[i]];
        pwin[i] <- inv_logit(xb[i]);
    }
}
model {
    team ~ normal(0, sigma_team);
    home ~ normal(0.1, sigma_home);
    win ~ bernoulli(pwin);
}
generated quantities {
    real xb_n [Nnew];
    real pwin_n [Nnew];
    
    for(i in 1:Nnew) {
        xb_n[i] <- home + team[tidnew[i]] - team[oidnew[i]];
        pwin_n[i] <- inv_logit(xb_n[i]);
    }
}
"""

data_simple = {
    "nteams": 32,
    "N": len(traingames),
    "win": traingames.win,
    "tid": traingames.tid+1,
    "oid": traingames.oid+1,
    "Nnew": len(testgames),
    "tidnew": testgames.tid+1,
    "oidnew": testgames.oid+1,
}

niter_simple = 1000
fit_simple = ps.stan(model_code=stan_simple, data=data_simple, iter=niter_simple, chains=2)

def team_scores(fit, team, var="team"):
    return sorted(fit.extract()[var][:, team_id(team)])

alpha=0.25

canvas = tp.Canvas()
axes = canvas.cartesian(label="Team qualities", xlabel="Score", ylabel="Team", xmin=-2.8)

data = [(idx, team, np.mean(team_scores(fit_simple, team)), team_scores(fit_simple, team)) 
        for idx, team in id_to_team.items()]

data.sort(key=lambda x: x[2])

axes.hlines([x - 0.5 for x in range(1,32)], opacity=0.1)
axes.vlines([0], opacity=0.1)

for idx, (_, _, _, scores) in enumerate(data):
    axes.plot([scores[int(alpha*niter_simple)], scores[int((1-alpha)*niter_simple)]], [idx, idx])

ids = list(range(32))
_, teams, avg_scores, _ = zip(*data)
axes.scatterplot(avg_scores, ids, color="grey")

axes.y.ticks.locator = tp.locator.Explicit(labels=teams)
axes.y.ticks.labels.angle = 270
axes.y.spine.show = False

posterior_preds = fit_simple.extract()["pwin"]
posterior_preds_test = fit_simple.extract()["pwin_n"]
testgames["pwin"] = np.mean(posterior_preds_test, 0)

testgames[["team", "opp", "pwin"]]

canvas = tp.Canvas(500, 300)
axes = canvas.cartesian()

axes.bars(np.histogram(posterior_preds[:,-1], bins=50))

stan_scores = """
data {
    int<lower=0> nteams; // number of teams
    int<lower=0> N; // number of observations
    
    vector[N] ptsf; // Points for
    vector[N] ptsa; // Points against
    
    int tid[N]; // Team
    int oid[N]; // Opponent
    real at_home[N]; // Indicator for at home games
    
    int Nnew; // new observations
    int tid_n[Nnew];
    int oid_n[Nnew];
    real at_home_n[Nnew];
    
    real<lower=0> df;
}
transformed data {}
parameters {
    real intercept;
    real home_offense;
    real home_defense;
    
    vector[nteams] offense;
    vector[nteams] defense;
    
    real mu_home;
    real<lower=1> sigma_home;
    
    real<lower=1> sigma_offense;
    real<lower=1> sigma_defense;
    
    real<lower=1> sigma_y;
}
transformed parameters {
    vector[N] xoff;
    vector[N] xdef;
    
    for(i in 1:N) {
        xoff[i] <- intercept + home_offense * at_home[i] + offense[tid[i]] - defense[oid[i]];
        xdef[i] <- intercept - home_defense * at_home[i] + offense[oid[i]] - defense[tid[i]];
    }
}
model {
    intercept ~ normal(22, 10);
    
    home_offense ~ normal(mu_home, sigma_home);
    home_defense ~ normal(mu_home, sigma_home);
    
    offense ~ normal(0, sigma_offense);
    defense ~ normal(0, sigma_defense);

    for(i in 1:N) {
        ptsf[i] ~ student_t(df, xoff, sigma_y);
        ptsa[i] ~ student_t(df, xdef, sigma_y);
    }
}
generated quantities {
    vector[Nnew] xoff_n;
    vector[Nnew] xdef_n;
    vector[Nnew] ptsf_n;
    vector[Nnew] ptsa_n;
    
    for(i in 1:Nnew) {
        xoff_n[i] <- intercept + home_offense * at_home_n[i] + offense[tid_n[i]] - defense[oid_n[i]];
        xdef_n[i] <- intercept - home_defense * at_home_n[i] + offense[oid_n[i]] - defense[tid_n[i]];
        
        ptsf_n[i] <- student_t_rng(df, xoff_n[i], sigma_y);
        ptsa_n[i] <- student_t_rng(df, xdef_n[i], sigma_y);
    }
}
"""

data_scores = {
    "nteams": 32,
    "N": len(traingames),
    "ptsf": traingames.ptsf,
    "ptsa": traingames.ptsa,
    "tid": traingames.tid+1,
    "oid": traingames.oid+1,
    "at_home": traingames.at_home,
    "Nnew": len(testgames),
    "tid_n": testgames.tid+1,
    "oid_n": testgames.oid+1,
    "at_home_n": testgames.at_home,
    "df": 15
}

fit_scores = ps.stan(model_code=stan_scores, data=data_scores, iter=500, chains=2)

posterior_preds_ptsf = fit_scores.extract()["ptsf_n"]
posterior_preds_ptsa = fit_scores.extract()["ptsa_n"]

canvas = tp.Canvas(500, 300)
axes = canvas.cartesian()

axes.scatterplot(posterior_preds_ptsf[:,1], posterior_preds_ptsa[:,1])



