import findspark
findspark.init(r'E:\progs.install\spark-2.2.0-bin-hadoop2.7')

import pyspark
sc = pyspark.SparkContext(appName="myAppName")
from pyspark.sql import SQLContext
sqlc = SQLContext(sc)

sdf = sqlc.read.csv('test-data/input/test-data-input-csv-clean-*.csv', inferSchema=False, header=True)
sdf.toPandas()

sdf = sqlc.read.csv('test-data/input/test-data-input-csv-colmismatch-*.csv', inferSchema=False, header=True)
sdf.toPandas()

sdf = sqlc.read.csv('test-data/input/test-data-input-csv-reorder-*.csv', inferSchema=False, header=True)
sdf.toPandas()

import glob
import d6tstack.combine_csv

cfg_fnames = list(glob.glob('test-data/input/test-data-input-csv-reorder-*.csv'))
c = d6tstack.combine_csv.CombinerCSV(cfg_fnames, all_strings=True)

# check columns
col_preview = c.preview_columns()
print('all columns equal?' , col_preview['is_all_equal'])
print('')
print('in what order do columns appear in the files?')
print('')
col_preview['df_columns_order'].reset_index(drop=True)

# out-of-core combining
c2 = d6tstack.combine_csv.CombinerCSVAdvanced(c, cfg_col_sel=c.col_preview['columns_all'])
c2.combine_save('test-data/output/test-combined.csv')

sdf = sqlc.read.csv('test-data/output/test-combined.csv', inferSchema=False, header=True)
sdf.toPandas()

# coming soon



