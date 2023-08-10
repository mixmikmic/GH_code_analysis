get_ipython().run_line_magic('load_ext', 'sparkmagic.magics')

get_ipython().run_line_magic('manage_spark', '')

get_ipython().run_cell_magic('spark', '-s testing -c sql -o localvar ', 'show tables')

get_ipython().run_cell_magic('spark', '-s testing', 'df = spark.loadFromMapRDB("/user/testuser/eth/all_transactions_table")\ndf.createOrReplaceTempView("all_transactions_view")\ndf.count()\n#df.printSchema()\n#sqlDF = spark.sql("SELECT * from all_transactions_view")\n#sqlDF.show()')

get_ipython().run_cell_magic('spark', '-s testing -c sql -o localvar', 'create table all_transactions_hive_table as select * from all_transactions_view')

get_ipython().run_cell_magic('spark', '-s testing -c sql -o localvar', 'select count(*) from all_transactions_hive_table')

get_ipython().run_cell_magic('spark', '', 'df = spark.loadFromMapRDB("/user/testuser/all_txs")\ndf.createOrReplaceTempView("all_txs_view2")\nsqlDF = spark.sql("SELECT * from all_txs_view2")\ndf.write.parquet("/user/testuser/all_parquet_x2")')

get_ipython().run_line_magic('manage_spark', '')

get_ipython().run_cell_magic('spark', '', 'sc.version')





