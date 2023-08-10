import sqlite3
import pandas as pd

db = "/Users/flavio.clesio/Documents/Github/database.sqlite"

with sqlite3.connect(db) as con:

    player_attributes = pd.read_sql_query("SELECT * FROM Player_Attributes", con)
    players = pd.read_sql_query("SELECT * FROM Player", con)
    leagues = pd.read_sql_query("SELECT * FROM League", con)
    match = pd.read_sql_query("SELECT * FROM Match", con)
    team = pd.read_sql_query("SELECT * FROM Team", con)
    team_attributes = pd.read_sql_query("SELECT * FROM Team_Attributes", con)
    country = pd.read_sql_query("SELECT * FROM Country", con)



