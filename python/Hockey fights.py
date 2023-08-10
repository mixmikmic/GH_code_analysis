import pandas as pd

df = pd.read_excel('../data/hockey-fights.xlsx', sheet_name='fights')

df.head()

df.info()

df.date.min()

df.date.max()

df.away_team_name.unique()

df.home_team_name.unique()

# etc ...

all_player_ids = pd.concat([df.home_player_id, df.away_player_id])

top = all_player_ids.value_counts().index[0]

player_record = df[df.away_player_id == top].iloc[0]
player_record

print(player_record.away_player_name, player_record.away_team_name)

df.shape

num_fights = df.shape[0]

num_games = df.game_id.nunique()

avg_fights_per_game = num_fights / num_games

print(avg_fights_per_game)

df['fight_duration'] = (df.fight_minutes * 60) + df.fight_seconds
df.sort_values('fight_duration', ascending=False).iloc[0]

