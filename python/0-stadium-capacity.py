import pandas as pd

cap = pd.read_csv('https://raw.githubusercontent.com/lenwood/MLB-Attendance/master/data/clean/MLB-Stadium-Capacity.csv')

cap.shape

cap.Year.sort_values().unique()

cap1516 = pd.read_csv('../data/MLB-stadium-capacity-15-16.csv')

cap1516.shape

stadium_capacities = pd.concat([cap, cap1516]).sort_values(['Team', 'Year']).reset_index(drop=True)

stadium_capacities.Team.unique()

stadium_capacities[stadium_capacities.Team == 'ANA']

stadium_capacities[stadium_capacities.Team == 'ARI']

stadium_capacities[stadium_capacities.Team == 'ATL']

stadium_capacities.loc[34, 'Capacity'] = 50528
stadium_capacities.loc[35:37, 'Capacity'] = 49714
stadium_capacities.loc[38:44, 'Capacity'] = 50096
stadium_capacities.loc[45:47, 'Capacity'] = 49743

stadium_capacities[stadium_capacities.Team == 'BAL']

stadium_capacities.loc[54, 'Capacity'] = 54017
stadium_capacities.loc[56:60, 'Capacity'] = 48041
stadium_capacities.loc[61:64, 'Capacity'] = 48079
stadium_capacities.loc[65:68, 'Capacity'] = 48190
stadium_capacities.loc[69:74, 'Capacity'] = 48290

stadium_capacities[stadium_capacities.Team == 'BOS']

stadium_capacities.loc[100, 'Capacity'] = 37400

stadium_capacities[stadium_capacities.Team == 'CAL']

stadium_capacities.loc[114, 'Capacity'] = 64593

stadium_capacities[stadium_capacities.Team == 'CHC']

stadium_capacities.loc[115:118, 'Capacity'] = 38711
stadium_capacities.loc[119:122, 'Capacity'] = 38765
stadium_capacities.loc[123:125, 'Capacity'] = 38884
stadium_capacities.loc[126, 'Capacity'] = 39059
stadium_capacities.loc[127:128, 'Capacity'] = 39111
stadium_capacities.loc[129, 'Capacity'] = 39345
stadium_capacities.loc[130, 'Capacity'] = 39538
stadium_capacities.loc[131, 'Capacity'] = 41118
stadium_capacities.loc[132:133, 'Capacity'] = 41160
stadium_capacities.loc[134:135, 'Capacity'] = 41210
stadium_capacities.loc[136, 'Capacity'] = 41159
stadium_capacities.loc[137, 'Capacity'] = 41009
stadium_capacities.loc[138, 'Capacity'] = 41019
stadium_capacities.loc[139, 'Capacity'] = 41072

stadium_capacities[stadium_capacities.Team == 'CHW']

stadium_capacities.loc[153, 'Capacity'] = 47522
stadium_capacities.loc[154:155, 'Capacity'] = 47098

stadium_capacities[stadium_capacities.Team == 'CIN']

stadium_capacities[stadium_capacities.Team == 'CLE']

stadium_capacities.loc[200:222, 'Stadium'] = 'Progressive Field'

stadium_capacities[stadium_capacities.Team == 'COL']

stadium_capacities[stadium_capacities.Team == 'DET']

stadium_capacities.loc[257:271, 'Stadium'] = 'Comerica Park'
stadium_capacities.loc[254:256, 'Capacity'] = 46945

stadium_capacities[stadium_capacities.Team == 'FLA']

stadium_capacities.drop([274, 275, 276], inplace=True)

stadium_capacities.loc[277, 'Capacity'] = 43909
stadium_capacities.loc[279, 'Capacity'] = 46238
stadium_capacities.loc[280:281, 'Capacity'] = 41855
stadium_capacities.loc[282, 'Capacity'] = 42531
stadium_capacities.loc[283, 'Capacity'] = 35521
stadium_capacities.loc[284:291, 'Capacity'] = 36331
stadium_capacities.loc[292:295, 'Capacity'] = 38560

stadium_capacities[stadium_capacities.Team == 'HOU']

stadium_capacities[stadium_capacities.Team == 'KCR']

stadium_capacities.loc[342:347, 'Capacity'] = 37903

stadium_capacities[stadium_capacities.Team == 'LAA']

stadium_capacities[stadium_capacities.Team == 'LAD']

stadium_capacities[stadium_capacities.Team == 'MIA']

stadium_capacities[stadium_capacities.Team == 'MIL']

stadium_capacities[stadium_capacities.Team == 'MIN']

stadium_capacities[stadium_capacities.Team == 'MON']

stadium_capacities[stadium_capacities.Team == 'NYM']

stadium_capacities[stadium_capacities.Team == 'NYY']

stadium_capacities[stadium_capacities.Team == 'OAK']

stadium_capacities[stadium_capacities.Team == 'PHI']

stadium_capacities[stadium_capacities.Team == 'PIT']

stadium_capacities[stadium_capacities.Team == 'SDP']

stadium_capacities[stadium_capacities.Team == 'SEA']

stadium_capacities.loc[625, 'Capacity'] = 58850
stadium_capacities.loc[626:628, 'Capacity'] = 57748

stadium_capacities[stadium_capacities.Team == 'SFG']

stadium_capacities[stadium_capacities.Team == 'STL']

stadium_capacities.loc[703, 'Capacity'] = 45399

stadium_capacities[stadium_capacities.Team == 'TBD']

stadium_capacities[stadium_capacities.Team == 'TBR']

stadium_capacities[stadium_capacities.Team == 'TEX']

stadium_capacities[stadium_capacities.Team == 'TOR']

stadium_capacities[stadium_capacities.Team == 'WSN']

stadium_capacities.loc[782:783, 'Capacity'] = 41888
stadium_capacities.loc[784:785, 'Capacity'] = 41546
stadium_capacities.loc[786, 'Capacity'] = 41487

stadium_capacities.to_csv('../data/MLB-stadium-capacity.csv', index=False, encoding='utf-8')

