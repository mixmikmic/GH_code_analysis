import datetime
print "Created on: {} ".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))

import os
import glob
import gspread
import warnings
import datetime
import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.externals import joblib

from sklearn.linear_model import LogisticRegression
pd.options.display.float_format = '{:,.2f}'.format
from sklearn import metrics

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

base_folder = os.path.abspath(os.getcwd())
os.chdir(".")
db_folder = os.getcwd() + "/new_data"
os.chdir(db_folder)

from oauth2client.service_account import ServiceAccountCredentials
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name('API Project-f22fe0b03992.json', scope)

main = pd.read_csv("1986_2016_seasons_shifted_v1.csv")
main.shape

main.wl_ta.value_counts(normalize=True)

for i in ['pts', 'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk', 'to']:
    main["{}_d".format(i)] = main['{}_ta'.format(i)] - main['{}_tb'.format(i)]

for i in ['pts', 'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk', 'to']:
    main["{}_ta_d".format(i)] = main['{}_ta'.format(i)] - main['{}_ta_opp'.format(i)]
    
for i in ['pts', 'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk', 'to']:
    main["{}_tb_d".format(i)] = main['{}_tb'.format(i)] - main['{}_tb_opp'.format(i)]

gc = gspread.authorize(credentials)
dashboard = gc.open("Tracking NBA Prediction Models").worksheet("Logistic Model")

def test_data_scores(row, model_name, model, model_specs, x, y, x_, y_):
    row_model = dashboard.find(str(row))._row
    model_inputs = dashboard.range('A{}:K{}'.format(row_model, row_model))
    y_pred = model.predict(x_)
    values = [row_model - 1, model_name,
              model.score(x, y), 
              model.score(x_, y_), 
              metrics.recall_score(y_, y_pred), 
              metrics.precision_score(y_, y_pred),
              metrics.f1_score(y_, y_pred), 
              metrics.roc_auc_score(y_, y_pred),
              metrics.log_loss(y_, y_pred),
              str(datetime.datetime.now()), "".join(model_specs.splitlines())]
    
    for cell, value in zip(model_inputs, values):
        cell.value = value        
    
    return dashboard.update_cells(model_inputs)

def for_analysis(dataframe, var):
    """This functions selects the variables required and make them into sklearn-ready formats! """
    y, x = dmatrices('wl_ta ~ ' + var, dataframe, return_type="dataframe")
    y = np.ravel(y)
    return y, x

model_9_diff_vars = '''
w_rate_ta * g_ta + w_rate_tb * g_tb + g_ta + g_tb + p_games_ta + p_games_tb + 
pts_ta + oreb_ta + dreb_ta + ast_ta + stl_ta + blk_ta + to_ta +
pts_tb + oreb_tb + dreb_tb + ast_tb + stl_tb + blk_tb + to_tb + 
pts_ta_opp + oreb_ta_opp + dreb_ta_opp + ast_ta_opp + stl_ta_opp + blk_ta_opp + to_ta_opp +
pts_tb_opp + oreb_tb_opp + dreb_tb_opp + ast_tb_opp + stl_tb_opp + blk_tb_opp + to_tb_opp +

pts_ta_d + oreb_ta_d + dreb_ta_d + ast_ta_d + stl_ta_d + blk_ta_d + to_ta_d +
pts_tb_d + oreb_tb_d + dreb_tb_d + ast_tb_d + stl_tb_d + blk_tb_d + to_tb_d + 

pts_d + oreb_d + dreb_d + ast_d + stl_d + blk_d + to_d + 

efg_ta + fgp_ta + efg_ta_opp + fgp_ta_opp + fta_fga_ta + fta_fga_ta_opp + fg3p_ta + ftp_ta + 
efg_tb + fgp_tb + efg_tb_opp + fgp_tb_opp + fta_fga_tb + fta_fga_tb_opp + fg3p_tb + ftp_tb +
C(team_id_ta)
'''

# y_test_advanced, x_test_advanced = for_analysis(main[main.season == main.season.max()], model_9_diff_vars)
# y_train_advanced, x_train_advanced = for_analysis(main[(main.season < main.season.max())], model_9_diff_vars)

# reg_logit = LogisticRegression(random_state=1984, C=0.01)
# reg_logit.fit(x_train_advanced, y_train_advanced)

# test_data_scores(15, "Mod_9+diff_vars", 
#                  model=reg_logit, model_specs = model_9_diff_vars, 
#                  x=x_train_advanced, y=y_train_advanced,
#                  x_=x_test_advanced, y_=y_test_advanced)

main["pts_ast_ta"] = main['pts_ta'] / main['ast_ta']
main["pts_ast_tb"] = main['pts_tb'] / main['ast_tb']

main["pts_ast_ta_opp"] = main['pts_ta'] / main['ast_ta']
main["pts_ast_tb_opp"] = main['pts_tb'] / main['ast_tb']

main['game_win_rates_ta'] = main["w_rate_ta"] * main['g_ta'] 
main['game_win_rates_tb'] = main["w_rate_tb"] * main['g_tb'] 

model_9_ratio_vars = '''
game_win_rates_ta + game_win_rates_tb + g_ta + g_tb + p_games_ta + p_games_tb + 
pts_ast_ta + pts_ast_tb + pts_ast_ta_opp + pts_ast_tb_opp + 

pts_ta + oreb_ta + dreb_ta + ast_ta + stl_ta + blk_ta + to_ta +
pts_tb + oreb_tb + dreb_tb + ast_tb + stl_tb + blk_tb + to_tb + 
pts_ta_opp + oreb_ta_opp + dreb_ta_opp + ast_ta_opp + stl_ta_opp + blk_ta_opp + to_ta_opp +
pts_tb_opp + oreb_tb_opp + dreb_tb_opp + ast_tb_opp + stl_tb_opp + blk_tb_opp + to_tb_opp +

efg_ta + fgp_ta + efg_ta_opp + fgp_ta_opp + fta_fga_ta + fta_fga_ta_opp + fg3p_ta + ftp_ta + 
efg_tb + fgp_tb + efg_tb_opp + fgp_tb_opp + fta_fga_tb + fta_fga_tb_opp + fg3p_tb + ftp_tb
'''

y_test_advanced, x_test_advanced = for_analysis(main, model_9_ratio_vars)
y_train_advanced, x_train_advanced = for_analysis(main, model_9_ratio_vars)

reg_logit = LogisticRegression(random_state=1984, C=0.01)
reg_logit.fit(x_train_advanced, y_train_advanced)

test_data_scores(26, "Full_Mod_9+ratio_vars", 
                 model=reg_logit, model_specs = model_9_ratio_vars, 
                 x=x_train_advanced, y=y_train_advanced,
                 x_=x_test_advanced, y_=y_test_advanced)

os.getcwd()

filename = 'First_production_model.sav'
joblib.dump(reg_logit, filename)

