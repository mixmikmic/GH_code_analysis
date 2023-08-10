import dask.dataframe as dd

# consistent format
ddf = dd.read_csv('test-data/input/test-data-input-csv-clean-*.csv')
ddf.compute()

# consistent format
ddf = dd.read_csv('test-data/input/test-data-input-csv-colmismatch-*.csv')
ddf.compute()

import glob
import d6tstack.combine_csv

cfg_fnames = list(glob.glob('test-data/input/test-data-input-csv-colmismatch-*.csv'))
c = d6tstack.combine_csv.CombinerCSV(cfg_fnames, all_strings=True)

# check columns
print('all equal',c.is_all_equal())
print('')
c.is_col_present()

# out-of-core combining
c2 = d6tstack.combine_csv.CombinerCSVAdvanced(c, cfg_col_sel=c.col_preview['columns_all'])
c2.combine_save('test-data/output/test-combined.csv')

# consistent format
ddf = dd.read_csv('test-data/output/test-combined.csv')
ddf.compute()

# consistent format
ddf = dd.read_csv('test-data/input/test-data-input-csv-reorder-*.csv')
ddf.compute()

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

# consistent format
ddf = dd.read_csv('test-data/output/test-combined.csv')
ddf.compute()



