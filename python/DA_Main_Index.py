import pandas as pd

main = pd.read_csv('criminal_main.csv',encoding="latin-1" )

da = pd.read_csv('da_last_name_dictionary.csv',names=['DistrictAttorney','da_last_name'])

clean_da = pd.read_csv('clean_da.csv')

main.head()

index_table = main[['File','DistrictAttorney']]

index_table.head()

da.head()

merged_keys = index_table.merge(da,how='left',on='DistrictAttorney')

merged_keys.head()

clean_da.head()

clean_da_indexed = merged_keys.merge(clean_da, how='left', on='da_last_name')

clean_da_indexed.head()

clean_da_indexed.to_csv('DA_keys.csv')

