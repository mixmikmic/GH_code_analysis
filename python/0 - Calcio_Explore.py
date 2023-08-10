import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from pylab import *
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split

df_cal = pd.read_csv("../data/serieA_1516.csv")

inter      = df_cal[(df_cal['HomeTeam']=='Inter') | (df_cal['AwayTeam']=='Inter')]
inter_home = df_cal[df_cal['HomeTeam']=='Inter']
inter_away = df_cal[df_cal['AwayTeam']=='Inter']

teams = np.array(inter_home["AwayTeam"])
teams2 = np.array(inter_away["HomeTeam"])

#print teams
#print teams2

teams_a = [] 

for i,cal in enumerate(df_cal.iterrows()):
    teams_a.append(cal[1]["HomeTeam"])
    teams_a.append(cal[1]["AwayTeam"])

    if i == 9: break
        
print teams_a 

#teams 2015/16
#teams_a = ['Lazio', 'Verona', 'Empoli', 'Fiorentina', 'Frosinone', 'Inter', 'Juventus', 'Palermo', 'Sampdoria', 'Sassuolo',
#           'Bologna', 'Roma', 'Chievo', 'Milan', 'Torino', 'Atalanta', 'Udinese', 'Genoa', 'Carpi', 'Napoli']

#teams 2014/15
#teams_a = ['Lazio', 'Verona', 'Empoli', 'Fiorentina', 'Parma', 'Inter', 'Juventus', 'Palermo', 'Sampdoria', 'Sassuolo', 
#           'Cagliari', 'Roma', 'Chievo', 'Milan', 'Torino', 'Atalanta', 'Udinese', 'Genoa', 'Cesena', 'Napoli']

team_stat = []
team_stat_sum = []
team_stat_var = []
team_stat_var2 = []
team_stat_acc = []
team_stat_s = []
team_stat_st = []
team_stat_agg = []
team_stat_sc = []
team_stat_stc = []
team_stat_points = []

for team_a in teams_a:
    #print team_a
    team      = df_cal[(df_cal['HomeTeam']==team_a) | (df_cal['AwayTeam']==team_a)]
    team_home = df_cal[df_cal['HomeTeam']==team_a]
    team_away = df_cal[df_cal['AwayTeam']==team_a]
    #calculate points
    cond   = [(team["HomeTeam"]==team_a) & (team["FTHG"]>team["FTAG"]) | (team["AwayTeam"]==team_a) & (team["FTHG"]<team["FTAG"]),
              (team["HomeTeam"]==team_a) & (team["FTHG"]<team["FTAG"]) | (team["AwayTeam"]==team_a) & (team["FTHG"]>team["FTAG"]),
              team["FTHG"]==team["FTAG"]]
    #team['points'] = np.select(cond,choice)
    
    team_h_win = len(team_home[team_home['FTHG']>team_home['FTAG']])
    team_a_win = len(team_away[team_away['FTAG']>team_away['FTHG']])
    team_draw = len(team[team['FTAG']==team['FTHG']])
    team_points = 3*team_a_win + 3*team_h_win + team_draw
    #print team_home
    #print team_away
    #shots made
    team_s    = team_away["AS"].sum()  + team_home["HS"].sum()
    team_st   = team_away["AST"].sum() + team_home["HST"].sum()
    team_acc  = float(team_st)/float(team_s)
    #shots conceded
    team_sc    = team_away["HS"].sum()  + team_home["AS"].sum()
    team_stc   = team_away["HST"].sum() + team_home["AST"].sum()
    team_agg  = float(team_stc)/float(team_sc)
    
    #print "ale s", team_s, 
    #print "ale st", team_st
    #print "ale acc", team_acc
    #calculate the shoot accuracy (one variable)    
    team_sh_ratio_a = team_away["AST"]/team_away["AS"]
    team_sh_ratio_h = team_home["HST"]/team_home["HS"]
    #print team_sh_ratio_a
    
    #calculate the metric number 0 (one variable)
    team_var_ratio_a = team_away["AS"]/team_away["HS"]
    team_var_ratio_h = team_home["HS"]/team_home["AS"]
    #calculate the metric number 0.5 (one variable)
    team_var2_ratio_a = team_away["AST"]/team_away["HST"]
    team_var2_ratio_h = team_home["HST"]/team_home["AST"]
    
    #calculate the metric number 1 (multiply)
    team_all_ratio_a = team_away["AST"]/team_away["AS"]*team_away["HS"]/team_away["HST"]
    team_all_ratio_h = team_home["HST"]/team_home["HS"]*team_home["AS"]/team_home["AST"]
    #calculate the metric number 1 (sum)
    team_all_ratio_sum_a = team_away["AST"]/team_away["AS"] + team_away["HS"]/team_away["HST"]
    team_all_ratio_sum_h = team_home["HST"]/team_home["HS"] + team_home["AS"]/team_home["AST"]
    #print "home ", np.mean(np.ma.masked_invalid(team_all_ratio_h))
    #print "away ",np.mean(np.ma.masked_invalid(team_all_ratio_a))
    team_stat.append(( team_a,
                       team_points, 
                       np.mean(np.ma.masked_invalid(team_all_ratio_a)) + np.mean(np.ma.masked_invalid(team_all_ratio_h))  ))
    team_stat_sum.append(( team_a, 
                           team_points, 
                           np.mean(np.ma.masked_invalid(team_all_ratio_sum_a)) + np.mean(np.ma.masked_invalid(team_all_ratio_sum_h))  ))
    team_stat_var.append(( team_a,
                           team_points,
                           np.mean(np.ma.masked_invalid(team_var_ratio_a)) + np.mean(np.ma.masked_invalid(team_var_ratio_h)),
                           np.std(np.ma.masked_invalid(team_var_ratio_a))/2 + np.std(np.ma.masked_invalid(team_var_ratio_h))/2  ))
    team_stat_var2.append(( team_a,
                            team_points,
                            np.mean(np.ma.masked_invalid(team_var2_ratio_a)) + np.mean(np.ma.masked_invalid(team_var2_ratio_h)),
                            np.std(np.ma.masked_invalid(team_var2_ratio_a))/2 + np.std(np.ma.masked_invalid(team_var2_ratio_h))/2  ))
    team_stat_points.append((team_a,
                            team_points))
    team_stat_acc.append(( team_a,
                          team_points, 
                          team_acc  ))
    team_stat_s.append(( team_a,
                          team_points, 
                          team_s  ))
    team_stat_st.append(( team_a,
                          team_points, 
                          team_st  ))
    team_stat_agg.append(( team_a,
                          team_points, 
                          team_agg  ))
    team_stat_sc.append(( team_a,
                          team_points, 
                          team_sc  ))
    team_stat_stc.append(( team_a,
                          team_points, 
                          team_stc  ))
    #print np.std(np.ma.masked_invalid(team_all_ratio_h))
    print team_a, team_points
    
team_stat_sort = sorted(team_stat, key=lambda x: x[1])
#print team_stat_sort

team_stat_sum_sort = sorted(team_stat_sum, key=lambda x: x[1])
#print team_stat_sum_sort

team_stat_var_sort = sorted(team_stat_var, key=lambda x: x[1])
#print team_stat_var_sort

team_stat_var2_sort = sorted(team_stat_var2, key=lambda x: x[1])
print team_stat_var2_sort
#df_cal

team_stat_points_sort = sorted(team_stat_points, key=lambda x: x[1])
print team_stat_points_sort

team_stat_acc_sort = sorted(team_stat_acc, key=lambda x: x[1])
print team_stat_acc_sort

team_stat_s_sort = sorted(team_stat_s, key=lambda x: x[1])
print team_stat_s_sort

team_stat_st_sort = sorted(team_stat_st, key=lambda x: x[1])
print team_stat_st_sort

team_stat_agg_sort = sorted(team_stat_agg, key=lambda x: x[1])
print team_stat_agg_sort

team_stat_sc_sort = sorted(team_stat_sc, key=lambda x: x[1])
print team_stat_sc_sort

team_stat_stc_sort = sorted(team_stat_stc, key=lambda x: x[1])
print team_stat_stc_sort

df_cal[['HomeTeam', 'AwayTeam', 'AS', 'AST', 'HS', 'HST']]

team_name    = [t[0] for t in team_stat_sort]
team_val   = [t[1] for t in team_stat_sort]

team_pos   = arange(len(team_name))+.5 

pos = arange(len(team_name))+.5    # the bar centers on the y axis

figure(1)
figure(figsize=(9,9))
barh(team_pos,team_val, align='center',color='r')
yticks(team_pos, team_name)
xlabel('Scoring metric Mult')
title('Scoring Metric Mult for Serie A teams')
grid(True)



team_name_sum    = [t[0] for t in team_stat_sum_sort]
team_val_sum   = [t[2] for t in team_stat_sum_sort]
team_pos_sum   = arange(len(team_name))+.5 

pos_sum = arange(len(team_name))+.5    # the bar centers on the y axis

figure(1)
figure(figsize=(9,9))
barh(team_pos_sum,team_val_sum, align='center',color='m')
yticks(team_pos_sum, team_name_sum)
xlabel('Scoring metric Sum')
title('Scoring Metric Sum for Serie A teams')
grid(True)

team_name_var    = [t[0] for t in team_stat_var_sort]
team_val_var   = [t[2] for t in team_stat_var_sort]
team_val_err   = [t[3] for t in team_stat_var_sort]

team_pos_var   = arange(len(team_name))+.5 

pos_sum = arange(len(team_name))+.5    # the bar centers on the y axis


figure(1)
figure(figsize=(9,9))
barh(team_pos_var,team_val_var, align='center',color='c')
yticks(team_pos_var, team_name_var)
xlabel('Scoring metric Var')
title('Scoring Metric Var for Serie A teams')
grid(True)

team_name_var2    = [t[0] for t in team_stat_var2_sort]
team_val_var2   = [t[2] for t in team_stat_var2_sort]
team_val_err2   = [t[3] for t in team_stat_var2_sort]

team_pos_var2   = arange(len(team_name))+.5 

pos_sum = arange(len(team_name))+.5    # the bar centers on the y axis


figure(1)
figure(figsize=(9,9))
barh(team_pos_var2,team_val_var2, align='center',color='k')
yticks(team_pos_var2, team_name_var2)
xlabel('Scoring metric Var')
title('Scoring Metric Var for Serie A teams')
grid(True)


team_name_sh    = [t[0] for t in team_stat_acc_sort]
team_val_sh   = [t[2] for t in team_stat_acc_sort]
team_pos_sh   = arange(len(team_name))+.5 
pos_sum = arange(len(team_name))+.5    # the bar centers on the y axis

figure(1)
figure(figsize=(9,9))
barh(team_pos_sh,team_val_sh, align='center',color='y')
yticks(team_pos_sh, team_name_sh)
xlabel('Scoring metric Shoot Accuracy')
title('Shoot Accuracy for Serie A teams')
grid(True)


team_name_sh    = [t[0] for t in team_stat_s_sort]
team_val_sh   = [t[2] for t in team_stat_s_sort]
team_pos_sh   = arange(len(team_name))+.5 
pos_sum = arange(len(team_name))+.5    # the bar centers on the y axis

figure(1)
figure(figsize=(9,9))
barh(team_pos_sh,team_val_sh, align='center',color='#ff1493')
yticks(team_pos_sh, team_name_sh)
xlabel('Scoring metric Total Shots')
title('Total Shots for Serie A teams')
grid(True)

team_name_var2    = [t[0] for t in team_stat_var2_sort]
team_val_var2     = [t[2] for t in team_stat_var2_sort]
team_val_points   = [t[1] for t in team_stat_var2_sort]

#define shooting accuracy
shoot_acc = team_val_var2

#print team_val_points
#team_pos_var2   = arange(len(team_name))+.5 

#pos_sum = arange(len(team_name))+.5    # the bar centers on the y axis


figure(1)
figure(figsize=(20,20))
fig, ax = plt.subplots(figsize=(10,10))

ax.scatter(team_val_var2,team_val_points,c='r',s=280,alpha=0.5)
for i, txt in enumerate(team_name_var2):
    ax.annotate(txt,(team_val_var2[i],team_val_points[i]))
xlabel('Scoring metric Var 2')
title('Scoring Metric Var for Serie A teams')
plt.show()


team_name_var    = [t[0] for t in team_stat_var_sort]
team_val_var     = [t[2] for t in team_stat_var_sort]
team_val_points   = [t[1] for t in team_stat_var_sort]

#print team_val_points
#team_pos_var2   = arange(len(team_name))+.5 

#pos_sum = arange(len(team_name))+.5    # the bar centers on the y axis


figure(1)
figure(figsize=(20,20))
fig, ax = plt.subplots(figsize=(10,10))

ax.scatter(team_val_var,team_val_points,c='c',s=280,alpha=0.5)
for i, txt in enumerate(team_name_var):
    ax.annotate(txt,(team_val_var[i],team_val_points[i]))
xlabel('Scoring metric Var')
title('Scoring Metric Var for Serie A teams')
plt.show()




team_name_var    = [t[0] for t in team_stat_s_sort]
team_val_var     = [t[2] for t in team_stat_s_sort]
team_val_points   = [t[1] for t in team_stat_s_sort]

figure(1)
figure(figsize=(20,20))
fig, ax = plt.subplots(figsize=(11,11))

ax.scatter(team_val_var,team_val_points,c='#ff1493',s=280,alpha=0.5)
for i, txt in enumerate(team_name_var):
    ax.annotate(txt,(team_val_var[i],team_val_points[i]))
xlabel('Total Shots')
title('Total Shots for Serie A teams')
plt.show()



team_name_var    = [t[0] for t in team_stat_st_sort]
team_val_var     = [t[2] for t in team_stat_st_sort]
team_val_points   = [t[1] for t in team_stat_st_sort]

figure(1)
figure(figsize=(20,20))
fig, ax = plt.subplots(figsize=(11,11))

ax.scatter(team_val_var,team_val_points,c='#9acd32',s=280,alpha=0.5)
for i, txt in enumerate(team_name_var):
    ax.annotate(txt,(team_val_var[i],team_val_points[i]))
xlabel('Total Shots On Target')
title('Total Shots On Target for Serie A teams')
plt.show()



team_name_var    = [t[0] for t in team_stat_acc_sort]
team_val_var     = [t[2] for t in team_stat_acc_sort]
team_val_points   = [t[1] for t in team_stat_acc_sort]

figure(1)
figure(figsize=(20,20))
fig, ax = plt.subplots(figsize=(11,11))

ax.scatter(team_val_var,team_val_points,c='#cd4f39',s=280,alpha=0.5)
for i, txt in enumerate(team_name_var):
    ax.annotate(txt,(team_val_var[i],team_val_points[i]))
xlabel('Shooting Accuracy')
title('Accuracy for Serie A teams')
plt.show()

team_name_var    = [t[0] for t in team_stat_sc_sort]
team_val_var     = [t[2] for t in team_stat_sc_sort]
team_val_points   = [t[1] for t in team_stat_sc_sort]

figure(1)
figure(figsize=(20,20))
fig, ax = plt.subplots(figsize=(11,11))

ax.scatter(team_val_var,team_val_points,c='#ff1493',s=280,alpha=0.5)
for i, txt in enumerate(team_name_var):
    ax.annotate(txt,(team_val_var[i],team_val_points[i]))
xlabel('Total Shots Conceded')
title('Total Shots for Serie A teams')
plt.show()



team_name_var    = [t[0] for t in team_stat_stc_sort]
team_val_var     = [t[2] for t in team_stat_stc_sort]
team_val_points   = [t[1] for t in team_stat_stc_sort]

figure(1)
figure(figsize=(20,20))
fig, ax = plt.subplots(figsize=(11,11))

ax.scatter(team_val_var,team_val_points,c='#9acd32',s=280,alpha=0.5)
for i, txt in enumerate(team_name_var):
    ax.annotate(txt,(team_val_var[i],team_val_points[i]))
xlabel('Total Shots On Target Conceded')
title('Total Shots On Target for Serie A teams')
plt.show()



team_name_var    = [t[0] for t in team_stat_agg_sort]
team_val_var     = [t[2] for t in team_stat_agg_sort]
team_val_points   = [t[1] for t in team_stat_agg_sort]

figure(1)
figure(figsize=(20,20))
fig, ax = plt.subplots(figsize=(11,11))

ax.scatter(team_val_var,team_val_points,c='#cd4f39',s=280,alpha=0.5)
for i, txt in enumerate(team_name_var):
    ax.annotate(txt,(team_val_var[i],team_val_points[i]))
xlabel('Shooting Aggressivity')
title('Accuracy for Serie A teams')
plt.show()

shoot_agg = team_val_var

team_feat = [[t,v] for t,v in zip(shoot_acc,shoot_agg)]
team_val_points


print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause

# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(team_feat, team_val_points, test_size=0.50, random_state=42)

print np.array(X_test)[:,0]
# Create linear regression object
regr = linear_model.LinearRegression()

print len(X_train)
print len(y_train)
# Train the model using the training sets
regr.fit(X_train, y_train)


# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))

# Plot outputs
plt.scatter(np.array(X_test)[:,0], y_test,  color='black')
plt.plot(X_test, regr.predict(X_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()



