import intake

intake.open_mapd # check that intake-mapd plugin is registered

if 0:
    # Using pymapd interface to learn what is in the test database, otherwise disable this part
    import pymapd
    con = pymapd.connect(user="mapd", password="HyperInteractive", host="localhost", dbname="mapd")
    print('con=',con)
    table_name = con.get_tables()[0]
    print('; '.join(['{0}'.format(d.name, d.type) for d in con.get_table_details(table_name)]))
    field_names = [d.name for d in con.get_table_details(table_name)]
    q = con.execute('SELECT {} FROM {};'.format(', '.join(field_names), table_name))
    print()
    print(q.fetchmany(1))

if 1: # Manually specify query information
    table_name = 'flights_2008_10k'
    field_names = ['carrier_name', 'dep_timestamp', 'origin', 'arr_timestamp', 'dest', 'airtime']

MAPD_URI='mapd://mapd:HyperInteractive@localhost:9091/mapd'

datasource = intake.open_mapd(MAPD_URI, table_name, field_names)  # Intake DataSource instance

datasource.discover()  # 

df = datasource.read() # reads all data to memory, df is pandas.DataFrame

df.head()              # show the head

