#Delete all sql tables
cursor = db.cursor()
cursor.execute(''' DROP table players ''')
cursor.execute(''' DROP table combine ''')
cursor.execute(''' DROP table rr ''')
cursor.execute(''' DROP table passing ''')

#Create sqlite database
import csv, sqlite3
db = sqlite3.connect(':memory:')
db = sqlite3.connect('nflPPdb')

cursor = db.cursor()
cursor.execute('''
    CREATE TABLE players( name varchar(50) PRIMARY KEY, 
                        college varchar(50),
                        draft_team varchar(50),
                        draft_round varchar(50),
                        draft_pick varchar(50),
                        draft_year int,
                        position varchar(50),
                        height varchar(50),
                        weight int)
    ''')
cursor.execute('''
    CREATE TABLE combine(year int,
                        name varchar(50) PRIMARY KEY,
                        firstname varchar(50),
                        lastname varchar(50),
                        position varchar(50),
                        heightfeet int,
                        heightinches int,
                        heightinchestotal int,
                        weight int,
                        arms int,
                        hands int,
                        fortyyd int,
                        twentyyd int,
                        tenyd int,
                        twentyss int,
                        threecone int,
                        vertical int,
                        broad int,
                        bench int,
                        round int,
                        college varchar(50),
                        pick int,
                        pickround int,
                        picktotal int)
    ''')
cursor.execute('''
    CREATE TABLE rr(name varchar(50),
                    year int,
                    team varchar(50),
                    age int,
                    position varchar(50),
                    games_played int,
                    games_started int,
                    rushing_attempts int,
                    rushing_yards int,
                    rushing_TD int,
                    rushing_long int,
                    rushing_ydsAtt float,
                    rushing_ydsGame float,
                    rushing_attGame float,
                    receiving_targets int,
                    receiving_receptions int,
                    receiving_yards int,
                    receiving_TD int,
                    receiving_long int,
                    receiving_recsGame float,
                    receiving_ydsGame float,
                    yardsfromScrimmage int,
                    RRTD int,
                    fumbles int,
                    PRIMARY KEY (name, year))
    ''')
cursor.execute('''
    CREATE TABLE passing(  name varchar(50),
                        year int,
                        team varchar(50),
                        age int,
                        position varchar(50),
                        games_played int,
                        games_started int,
                        wins int,
                        losses int,
                        completions int,
                        attempts int,
                        completionPct float,
                        passing_yards int,
                        passing_TD int,
                        passing_TDPct float,
                        passing_INT int,
                        passing_INTPct float,
                        passing_long int,
                        passing_ydsAtt float,
                        passing_airydsAtt float,
                        passing_ydsComp float,
                        passing_ydsGame float,
                        passing_rating float,
                        passing_sacks int,
                        passing_sacksyds int,
                        passing_airnetydsAtt float,
                        passing_sackPct float,
                        FourthQtrComebacks int,
                        gamewinningdrives int,
                        PRIMARY KEY (name, year))
    ''')

db.commit()
db.close()

#Load csvs into sql tables
import pandas as pd
con = sqlite3.connect("nflPPdb.sqlite")
con.text_factory=str
df1 = pd.read_csv('NFLPlayersDatabase.csv')
df1 = df1[['name','college','draft_team','draft_round','draft_pick','draft_year','position','height','weight']]
df1.to_sql(name='players', if_exists='append', con=con)

df2 = pd.read_csv('NFLCombineStats1999-2015.csv')
df2 = df2[['year','name','firstname','lastname','position','heightfeet','heightinches','heightinchestotal',
          'weight','arms','hands','fortyyd','twentyyd','tenyd','twentyss','threecone','vertical','broad',
          'bench','round','college','pick','pickround','picktotal']]
df2.to_sql(name='combine', if_exists='append', con=con)

df3 = pd.read_csv('pfr-rushing-receiving/mergedRushingReceiving.csv')
df3 = df3[['name', 'year', 'team','age', 'position', 'games_played','games_started', 'rushing_attempts', 
          'rushing_yards', 'rushing_TD','rushing_long','rushing_ydsAtt', 'rushing_ydsGame', 'rushing_attGame',
          'receiving_targets','receiving_receptions','receiving_yards','receiving_TD','receiving_long',
          'receiving_recsGame','receiving_ydsGame','yardsfromScrimmage', 'RRTD', 'fumbles']]
df3.to_sql(name='rr', if_exists='append', con=con)
df4 = pd.read_csv('pfr-passing/mergedPassing.csv')
df4 = df4[['name','year','team','age','position','games_played','games_started','wins','losses','completions','attempts',
          'completionPct','passing_yards','passing_TD','passing_TDPct','passing_INT','passing_INTPct','passing_long',
          'passing_ydsAtt','passing_airydsAtt','passing_ydsComp','passing_ydsGame','passing_rating','passing_sacks',
          'passing_sacksyds','passing_airnetydsAtt','passing_sackPct','FourthQtrComebacks','gamewinningdrives']]
df4.to_sql(name='passing', if_exists='append', con=con)

#Convert SQL tables to pandas data frames
players = pd.read_sql_query('SELECT * FROM players', con)
combine = pd.read_sql_query('SELECT * FROM combine', con)
rr = pd.read_sql_query('SELECT * FROM rr', con)
passing = pd.read_sql_query('SELECT * FROM passing', con)

