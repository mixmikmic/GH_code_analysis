# loading the special sql extension
get_ipython().magic('load_ext sql')

# connecting to a database which lives on the Amazon Cloud
# need to substitute password with the one provided in the email!!!
get_ipython().magic('sql postgresql://dssg_student:password@seds-sql.csya4zsfb6y4.us-east-1.rds.amazonaws.com/dssg2016')

# running a simple SQL command
get_ipython().magic('sql select * from seattlecrimeincidents limit 10;')

# Show specific columns
get_ipython().magic('sql select "Offense Type",latitude,longitude from seattlecrimeincidents limit 10;')

get_ipython().run_cell_magic('sql', '', '-- select rows\nselect "Offense Type", latitude, longitude, month from seattlecrimeincidents\n    where "Offense Type" =\'THEFT-BICYCLE\' and month = 1')

get_ipython().run_cell_magic('sql', '', 'select count(*) from seattlecrimeincidents;')

get_ipython().run_cell_magic('sql', '', 'select count(*) from settlecrimeincidents')

get_ipython().run_cell_magic('sql', '', 'select count(*) from (select "Offense Type", latitude, longitude, month from seattlecrimeincidents\n    where "Offense Type" =\'THEFT-BICYCLE\' and month = 1) as small_table')

# use max, min functions

get_ipython().run_cell_magic('sql', '', 'select min(latitude) as min_lat,max(latitude) as max_lat,\n        min(longitude)as min_long,max(longitude) as max_long\n        from seattlecrimeincidents;')

get_ipython().run_cell_magic('sql', '', 'select year,count(*) from seattlecrimeincidents \n    group by year\n    order by year ASC;')

get_ipython().run_cell_magic('sql', '', 'select distinct year from seattlecrimeincidents;')







