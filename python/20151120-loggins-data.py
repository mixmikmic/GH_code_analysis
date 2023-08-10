from IPython.display import Image
Image(url="screenshot.png", height=600, width=400)

get_ipython().system('wc *.csv')

get_ipython().system('csvlook building.csv')

get_ipython().system('csvlook floor.csv')

get_ipython().system('csvlook zone.csv')

get_ipython().system('head -10 location.csv | csvcut -c1,2,3,4,5,10 | csvlook')

get_ipython().system('head -20 location.csv | csvcut -c1,6,7,8,9 | csvlook')

get_ipython().system('csvcut -c6 location.csv | csvsort -c1 | sort | uniq -c')

get_ipython().system('csvcut -c5 location.csv | sort | uniq -c')

get_ipython().system('head session.csv | csvlook')

get_ipython().system('csvcut -c5 session.csv | sort | uniq -c | head -2')

get_ipython().magic('load_ext sql')

get_ipython().system('createdb -U dchud logginsdw')

get_ipython().system('csvsql --db postgresql:///logginsdw --insert building.csv')

get_ipython().system('csvsql --db postgresql:///logginsdw --insert floor.csv')

get_ipython().system('csvsql --db postgresql:///logginsdw --insert zone.csv')

get_ipython().system('csvsql --db postgresql:///logginsdw --insert location.csv')

get_ipython().system('csvsql --db postgresql:///logginsdw --insert session.csv')

get_ipython().magic('sql postgresql://dchud@localhost/logginsdw')

get_ipython().run_cell_magic('sql', '', 'SELECT * FROM zone')

get_ipython().run_cell_magic('sql', '', 'SELECT * FROM location LIMIT 5')

get_ipython().run_cell_magic('sql', '', 'SELECT * FROM session LIMIT 5')

Image(url="star-schema.jpeg", height=600, width=400)

Image(url="agile-design.jpg", height=600, width=400)

get_ipython().run_cell_magic('sql', '', "DROP TABLE IF EXISTS d_location;\nCREATE TABLE d_location (\n  location_key SERIAL NOT NULL PRIMARY KEY,\n  building_name CHAR(6) NOT NULL,\n  building_id SMALLINT NOT NULL DEFAULT 0,\n  floor SMALLINT NOT NULL,  \n  floor_id SMALLINT NOT NULL DEFAULT 0,\n  zone_name CHAR(25) NOT NULL DEFAULT '',\n  zone_display_order SMALLINT NOT NULL,\n  zone_id SMALLINT NOT NULL DEFAULT 0,\n  location_id SMALLINT NOT NULL\n)")

get_ipython().run_cell_magic('sql', '', 'DELETE FROM d_location;\nINSERT INTO d_location (building_name, building_id,\n                       floor, floor_id,\n                       zone_name, zone_display_order, zone_id,\n                       location_id)\nSELECT building.name AS building_name, building.id AS building_id,\n       floor.floor AS floor, floor.id AS floor_id,\n       zone.name AS zone_name, zone.display_order AS zone_display_order, zone.id AS zone_id,\n       location.id\nFROM location, zone, floor, building\nWHERE location.zone_id = zone.id\n  AND zone.floor_id = floor.id\n  AND floor.id = zone.floor_id\n  AND building.id = floor.building_id;')

get_ipython().run_cell_magic('sql', '', "INSERT INTO d_location (building_name, building_id, floor, floor_id, \n                        zone_name, zone_display_order, zone_id,\n                        location_id)\nVALUES ('None', 0, 0, 0, 'None', 0, 0, 0)")

get_ipython().run_cell_magic('sql', '', 'SELECT * FROM d_location;')

get_ipython().run_cell_magic('sql', '', 'CREATE INDEX idx_location_id ON d_location (location_id)')

get_ipython().run_cell_magic('sql', '', 'DROP TABLE IF EXISTS d_day;\nCREATE TABLE d_day (\n  day_key SERIAL NOT NULL PRIMARY KEY,\n  full_date DATE NOT NULL,\n  day SMALLINT NOT NULL,\n  day_of_week_number SMALLINT NOT NULL,\n  day_of_week_name CHAR(9) NOT NULL,\n  day_of_week_abbr CHAR(3) NOT NULL,\n  day_of_year SMALLINT NOT NULL,\n  week_of_month SMALLINT NOT NULL,\n  week_of_year SMALLINT NOT NULL,\n  federal_holiday_flag BOOLEAN DEFAULT FALSE,\n  gwu_holiday_flag BOOLEAN DEFAULT FALSE,\n  gwu_in_session BOOLEAN DEFAULT FALSE,\n  weekday_flag BOOLEAN,\n  weekend_flag BOOLEAN,\n  month SMALLINT NOT NULL,\n  month_name CHAR(9) NOT NULL,\n  month_abbr CHAR(3) NOT NULL,\n  quarter SMALLINT NOT NULL,\n  year SMALLINT NOT NULL\n)')

get_ipython().run_cell_magic('sql', '', 'SELECT * FROM session LIMIT 5;')

get_ipython().run_cell_magic('sql', '', 'CREATE INDEX idx_timestamp_start ON session (timestamp_start);')

get_ipython().run_cell_magic('sql', '', "DELETE FROM d_day;\nINSERT INTO d_day\n  (full_date, \n   day, day_of_week_number, day_of_week_name, day_of_week_abbr, day_of_year,\n   week_of_month, week_of_year,\n   federal_holiday_flag, gwu_holiday_flag, gwu_in_session,\n   weekday_flag, weekend_flag,\n   month, month_name, month_abbr,\n   quarter, year)\nSELECT DISTINCT DATE(timestamp_start) AS full_date,\n   TO_NUMBER(TO_CHAR(timestamp_start, 'DD'), '99') AS day_of_month,\n     TO_NUMBER(TO_CHAR(timestamp_start, 'D'), '9') AS day_of_week_number,\n     TO_CHAR(timestamp_start, 'Day') AS day_of_week_name,\n     TO_CHAR(timestamp_start, 'Dy') AS day_of_week_abbr,\n     TO_NUMBER(TO_CHAR(timestamp_start, 'DDD'), '999') AS day_of_year,\n   TO_NUMBER(TO_CHAR(timestamp_start, 'W'), '9') AS week_of_month,\n     TO_NUMBER(TO_CHAR(timestamp_start, 'WW'), '99') AS week_of_year,\n   FALSE AS federal_holiday_flag, \n     FALSE AS gwu_holiday_flag,\n     FALSE AS gwu_in_session,\n   TO_NUMBER(TO_CHAR(timestamp_start, 'ID'), '9') <= 5 AS weekday_flag,\n     TO_NUMBER(TO_CHAR(timestamp_start, 'ID'), '9') > 5 AS weekend_flag,\n   TO_NUMBER(TO_CHAR(timestamp_start, 'MM'), '99') AS month,\n     TO_CHAR(timestamp_start, 'Month') AS month_name,\n     TO_CHAR(timestamp_start, 'Mon') AS month_abbr,\n   TO_NUMBER(TO_CHAR(timestamp_start, 'Q'), '9') AS quarter,\n     TO_NUMBER(TO_CHAR(timestamp_start, 'YYYY'), '9999') AS year\nFROM session\nORDER BY full_date")

get_ipython().run_cell_magic('sql', '', 'SELECT * FROM d_day LIMIT 20;')

get_ipython().run_cell_magic('sql', '', 'CREATE INDEX idx_full_date ON d_day (full_date)')

get_ipython().run_cell_magic('sql', '', "DROP TABLE IF EXISTS f_login;\nCREATE TABLE f_login (\n  login_id SERIAL PRIMARY KEY,\n  location_key SMALLINT NOT NULL DEFAULT 15,\n  timestamp_start TIMESTAMP NOT NULL,\n  day_key_start SMALLINT NOT NULL,\n  time_start CHAR(8) NOT NULL,\n  timestamp_end TIMESTAMP NOT NULL,\n  day_key_end SMALLINT NOT NULL,\n  time_end CHAR(8) NOT NULL,\n  hour_start SMALLINT NOT NULL,\n  hour_end SMALLINT NOT NULL,\n  duration_minutes REAL NOT NULL,\n  duration_hours REAL NOT NULL,\n  session_type CHAR(13) NOT NULL DEFAULT 'Available',\n  os CHAR(4) NOT NULL DEFAULT '',\n  session_id INT NOT NULL,\n  location_id INT NOT NULL\n)")

get_ipython().run_cell_magic('sql', '', "DELETE FROM f_login;\nINSERT INTO f_login (location_key,\n                     timestamp_start, day_key_start, time_start,\n                     timestamp_end, day_key_end, time_end,\n                     hour_start, hour_end,\n                     duration_minutes, duration_hours,\n                     session_type, os,\n                     session_id, location_id\n                     )\nSELECT 15 AS location_key,\n  timestamp_start, \n    1 AS day_key_start,\n    TO_CHAR(timestamp_start, 'HH24:MI:SS') AS time_start,\n  timestamp_end,\n    2 AS day_key_end,\n    TO_CHAR(timestamp_end, 'HH24:MI:SS') AS time_end,\n  TO_NUMBER(TO_CHAR(timestamp_start, 'HH24'), '99') AS hour_start,\n    TO_NUMBER(TO_CHAR(timestamp_end, 'HH24'), '99') AS hour_end,\n  EXTRACT(EPOCH FROM (timestamp_end - timestamp_start)) / 60 AS duration_minutes,\n    EXTRACT(EPOCH FROM (timestamp_end - timestamp_start)) / 3600 AS duration_hours,\n  CASE WHEN session_type='i' THEN 'In use'\n       WHEN session_type='n' THEN 'Unavailable' END AS session_type,\n    CASE WHEN os IS NULL THEN ''\n         ELSE os END, \n  session.id AS session_id,\n    session.location_id AS location_id\nFROM session, location\nWHERE location.id = session.location_id\nORDER BY session_id")

get_ipython().run_cell_magic('sql', '', 'SELECT * FROM f_login LIMIT 10')

get_ipython().run_cell_magic('sql', '', 'UPDATE f_login\nSET day_key_start = d_day.day_key\nFROM d_day\nWHERE DATE(f_login.timestamp_start) = d_day.full_date')

get_ipython().run_cell_magic('sql', '', 'UPDATE f_login\nSET day_key_end = d_day.day_key\nFROM d_day\nWHERE DATE(f_login.timestamp_end) = d_day.full_date')

get_ipython().run_cell_magic('sql', '', 'UPDATE f_login\nSET location_key = d_location.location_key\nFROM d_location\nWHERE f_login.location_id = d_location.location_id')

get_ipython().run_cell_magic('sql', '', 'SELECT * FROM f_login LIMIT 10')

get_ipython().magic('matplotlib inline')

get_ipython().run_cell_magic('sql', '', "SELECT AVG(duration_minutes), os, building_name\nFROM f_login, d_location\nWHERE d_location.location_key = f_login.location_key\nAND duration_hours < 12\nGROUP BY os, building_name\nHAVING os IN ('mac', 'win7')")

_.bar()

get_ipython().run_cell_magic('sql', '', 'SELECT hour_start, AVG(duration_minutes)\nFROM f_login, d_day\nWHERE d_day.day_key = f_login.day_key_start\nAND duration_hours < 12\nGROUP BY hour_start\nORDER BY hour_start')

_.bar()

get_ipython().run_cell_magic('sql', '', 'SELECT COUNT(*), session_type, os\nFROM f_login\nWHERE duration_hours > 48\nGROUP BY os, session_type\nLIMIT 50')

_.bar()

get_ipython().run_cell_magic('sql', '', "SELECT os, session_type, ROUND(duration_minutes) AS minutes, COUNT(*)\nFROM f_login\nWHERE duration_minutes <= 20\nAND os = 'win7'\nGROUP BY os, session_type, minutes\nORDER BY os, session_type, minutes ASC")

_.bar()

get_ipython().run_cell_magic('sql', '', "SELECT os, session_type, ROUND(duration_minutes) AS minutes, COUNT(*)\nFROM f_login\nWHERE duration_minutes <= 20\nAND os = 'win7'\nAND session_type = 'Unavailable'\nGROUP BY os, session_type, minutes\nORDER BY os, session_type, minutes ASC")

_.bar()

get_ipython().run_cell_magic('sql', '', "SELECT os, session_type, ROUND(duration_minutes) AS minutes, COUNT(*)\nFROM f_login\nWHERE duration_minutes <= 30\nAND os = 'win7'\nAND session_type = 'In use'\nGROUP BY os, session_type, minutes\nORDER BY os, session_type, minutes ASC")

_.bar()

get_ipython().run_cell_magic('sql', '', "SELECT os, session_type, ROUND(duration_minutes) AS minutes, COUNT(*)\nFROM f_login\nWHERE duration_minutes <= 30\nAND os = 'mac'\nAND session_type = 'In use'\nGROUP BY os, session_type, minutes\nORDER BY os, session_type, minutes ASC")

_.bar()

get_ipython().run_cell_magic('sql', '', "SELECT os, session_type, ROUND(duration_minutes) AS minutes, COUNT(*)\nFROM f_login, d_location\nWHERE d_location.location_key = f_login.location_key\n  AND duration_minutes <= 30\n  AND os = 'win7'\n  AND session_type = 'In use'\n  AND building_name = 'Gelman'\n  AND floor = 2\nGROUP BY os, session_type, minutes\nORDER BY os, session_type, minutes ASC")

_.bar()

get_ipython().run_cell_magic('sql', '', "SELECT os, session_type, ROUND(duration_minutes) AS minutes, COUNT(*)\nFROM f_login, d_location\nWHERE d_location.location_key = f_login.location_key\n  AND duration_minutes <= 30\n  AND os = 'win7'\n  AND session_type = 'In use'\n  AND building_name = 'Gelman'\n  AND floor != 2\nGROUP BY os, session_type, minutes\nORDER BY os, session_type, minutes ASC")

_.bar()

get_ipython().run_cell_magic('sql', '', "SELECT os, session_type, ROUND(duration_minutes) AS minutes, COUNT(*)\nFROM f_login, d_location\nWHERE d_location.location_key = f_login.location_key\n  AND duration_minutes <= 30\n  AND os = 'win7'\n  AND session_type = 'In use'\n  AND building_name = 'Eckles'\nGROUP BY os, session_type, minutes\nORDER BY os, session_type, minutes ASC")

_.bar()

get_ipython().run_cell_magic('sql', '', "SELECT session_type, ROUND(duration_minutes) AS minutes, COUNT(*)\nFROM f_login, d_day\nWHERE d_day.day_key = f_login.day_key_start\n  AND duration_minutes <= 30\n  AND session_type = 'In use'\n  AND weekday_flag = True\nGROUP BY session_type, minutes\nORDER BY session_type, minutes ASC")

_.bar()

get_ipython().run_cell_magic('sql', '', "SELECT session_type, ROUND(duration_minutes) AS minutes, COUNT(*)\nFROM f_login, d_day\nWHERE d_day.day_key = f_login.day_key_start\n  AND duration_minutes <= 30\n  AND session_type = 'In use'\n  AND weekend_flag = True\nGROUP BY session_type, minutes\nORDER BY session_type, minutes ASC")

_.bar()

get_ipython().system('head -20 20150802-entrance.csv | csvlook')

get_ipython().system('csvsql --db postgresql:///logginsdw --insert entrance.csv')

get_ipython().run_cell_magic('sql', '', 'SELECT * FROM entrance LIMIT 10')

get_ipython().run_cell_magic('sql', '', 'DROP TABLE IF EXISTS f_entrance;\nCREATE TABLE f_entrance (\n  entrance_id SERIAL PRIMARY KEY,\n  day_key SMALLINT NOT NULL,\n  hour SMALLINT NOT NULL,\n  affiliation VARCHAR(255),\n  subaffiliation VARCHAR(255),\n  count INT NOT NULL\n)')

get_ipython().run_cell_magic('sql', '', "SELECT day_key \nFROM d_day \nWHERE full_date = '2015-08-02'")

get_ipython().run_cell_magic('sql', '', 'DELETE FROM f_entrance;\nINSERT INTO f_entrance\n  (day_key, hour, affiliation, subaffiliation, count)\nSELECT 713, hour, affiliation, subaffiliation, count\nFROM entrance\nORDER BY hour, affiliation, subaffiliation')

get_ipython().run_cell_magic('sql', '', 'SELECT * FROM f_entrance LIMIT 10')

get_ipython().run_cell_magic('sql', '', 'SELECT hour, SUM(COUNT)\nFROM f_entrance\nGROUP BY hour\nORDER BY hour')

_.bar()

get_ipython().run_cell_magic('sql', '', "SELECT hour_start, COUNT(*)\nFROM f_login, d_location\nWHERE d_location.location_key = f_login.location_key\n  AND building_name = 'Gelman'\n  AND day_key_start = 713\nGROUP BY hour_start\nORDER BY hour_start ASC")

get_ipython().run_cell_magic('sql', '', "SELECT hour_start, os, session_type, COUNT(*), ROUND(AVG(duration_minutes))\nFROM f_login, d_location\nWHERE d_location.location_key = f_login.location_key\n  AND building_name = 'Gelman'\n  AND day_key_start = 713\nGROUP BY hour_start, os, session_type\nORDER BY hour_start ASC")

get_ipython().run_cell_magic('sql', '', "SELECT hour_start, os, session_type, COUNT(*), ROUND(AVG(duration_minutes))\nFROM f_login, d_location\nWHERE d_location.location_key = f_login.location_key\n  AND building_name = 'Gelman'\n  AND day_key_start = 713\n  AND session_type = 'In use'\nGROUP BY hour_start, os, session_type\nORDER BY hour_start ASC")

_.bar()

get_ipython().run_cell_magic('sql', '', "SELECT hour_start, os, session_type, COUNT(*), ROUND(AVG(duration_minutes))\nFROM f_login, d_location\nWHERE d_location.location_key = f_login.location_key\n  AND building_name = 'Gelman'\n  AND day_key_start = 713\n  AND session_type = 'In use'\n  AND duration_minutes < 600\nGROUP BY hour_start, os, session_type\nORDER BY hour_start ASC")

_.bar()

get_ipython().run_cell_magic('sql', '', "SELECT hour_start AS hour, COUNT(*) AS total_sessions\nFROM f_login, d_location\nWHERE d_location.location_key = f_login.location_key\n  AND building_name = 'Gelman'\n  AND day_key_start = 713\n  AND session_type = 'In use'\nGROUP BY hour_start\nORDER BY hour_start ASC")

get_ipython().run_cell_magic('sql', '', 'SELECT hour, affiliation, subaffiliation, count AS group_entrances\nFROM f_entrance\nORDER BY hour ASC')

get_ipython().run_cell_magic('sql', '', "SELECT f.hour, f.affiliation, f.subaffiliation, f.count AS group_entrances, computer_sessions_started\nFROM f_entrance f\nLEFT JOIN\n(\n    SELECT hour_start AS hour, COUNT(*) AS computer_sessions_started\n    FROM f_login, d_location\n    WHERE d_location.location_key = f_login.location_key\n      AND building_name = 'Gelman'\n      AND day_key_start = 713\n      AND session_type = 'In use'\n    GROUP BY hour_start\n) s\n    ON s.hour = f.hour\nORDER BY f.hour, affiliation, subaffiliation ASC")

