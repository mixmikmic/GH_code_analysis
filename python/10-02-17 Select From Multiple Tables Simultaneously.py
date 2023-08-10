# Ignore
get_ipython().run_line_magic('load_ext', 'sql')
get_ipython().run_line_magic('sql', 'sqlite://')
get_ipython().run_line_magic('config', 'SqlMagic.feedback = False')

get_ipython().run_cell_magic('sql', '', "\n-- Create a table of criminals\nCREATE TABLE criminals (pid, name, age, sex, city, minor);\nINSERT INTO criminals VALUES (412, 'James Smith', 15, 'M', 'Santa Rosa', 1);\nINSERT INTO criminals VALUES (234, 'Bill James', 22, 'M', 'Santa Rosa', 0);\nINSERT INTO criminals VALUES (632, 'Stacy Miller', 23, 'F', 'Santa Rosa', 0);\nINSERT INTO criminals VALUES (621, 'Betty Bob', NULL, 'F', 'Petaluma', 1);\nINSERT INTO criminals VALUES (162, 'Jaden Ado', 49, 'M', NULL, 0);\nINSERT INTO criminals VALUES (901, 'Gordon Ado', 32, 'F', 'Santa Rosa', 0);\nINSERT INTO criminals VALUES (512, 'Bill Byson', 21, 'M', 'Santa Rosa', 0);\nINSERT INTO criminals VALUES (411, 'Bob Iton', NULL, 'M', 'San Francisco', 0);\n\n-- Create a table of crimes\nCREATE TABLE crimes (cid, crime, city, pid_arrested, cash_stolen);\nINSERT INTO crimes VALUES (1, 'fraud', 'Santa Rosa', 412, 40000);\nINSERT INTO crimes VALUES (2, 'burglary', 'Petaluma', 234, 2000);\nINSERT INTO crimes VALUES (3, 'burglary', 'Santa Rosa', 632, 2000);\nINSERT INTO crimes VALUES (4, NULL, NULL, 621, 3500); \nINSERT INTO crimes VALUES (5, 'burglary', 'Santa Rosa', 162, 1000); \nINSERT INTO crimes VALUES (6, NULL, 'Petaluma', 901, 50000); \nINSERT INTO crimes VALUES (7, 'fraud', 'San Francisco', 412, 60000); \nINSERT INTO crimes VALUES (8, 'burglary', 'Santa Rosa', 512, 7000); \nINSERT INTO crimes VALUES (9, 'burglary', 'San Francisco', 411, 3000); \nINSERT INTO crimes VALUES (10, 'robbery', 'Santa Rosa', 632, 2500); \nINSERT INTO crimes VALUES (11, 'robbery', 'Santa Rosa', 512, 3000);")

get_ipython().run_cell_magic('sql', '', '\nSELECT city FROM criminals\nUNION\nSELECT city FROM crimes;')

get_ipython().run_cell_magic('sql', '', 'SELECT city FROM criminals\nUNION ALL\nSELECT city FROM crimes;')

