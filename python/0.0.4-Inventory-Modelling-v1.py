from supplychainpy.model_inventory import analyse
from decimal import Decimal
from supplychainpy.sample_data.config import ABS_FILE_PATH

with open(ABS_FILE_PATH['COMPLETE_CSV_SM'],'r') as raw_data:
    for line in raw_data:
        print(line)

#%%timeit
analysed_inventory_profile= analyse(file_path=ABS_FILE_PATH['COMPLETE_CSV_SM'],
                                                             z_value=Decimal(1.28),
                                                             reorder_cost=Decimal(400),
                                                             file_type='csv')

analysis_summary = [demand.orders_summary() for demand in analysed_inventory_profile]
print(analysis_summary)

analysis_summary =[]
for demand in analysed_inventory_profile:
    analysis_summary.append(demand.orders_summary())

get_ipython().run_cell_magic('timeit', '', "sku_summary = [demand.orders_summary() for demand in analysed_inventory_profile if demand.orders_summary().get('sku')== 'KR202-209']")

ay_classification_summary = [demand.orders_summary() for demand in analysed_inventory_profile if demand.orders_summary().get('ABC_XYZ_Classification')== 'AY']
print(ay_classification_summary)

from supplychainpy.inventory.summarise import Inventory
filtered_summary = Inventory(analysed_inventory_profile)

get_ipython().run_cell_magic('timeit', '', "sku_summary = [summary for summary in filtered_summary.describe_sku('KR202-209')]\nprint(sku_summary)")

classification_summary =  [summary for summary in filtered_summary.abc_xyz_summary(classification=('AY',), category=('revenue',))]
print(classification_summary)

top_10_safety_stock_skus =  [summary.get('sku')for summary in filtered_summary.rank_summary(attribute='safety_stock', count=10)]
print(top_10_safety_stock_skus)

top_10_safety_stock_values =  [(summary.get('sku'), summary.get('safety_stock'))for summary in filtered_summary.rank_summary(attribute='safety_stock', count=10)]
print(top_10_safety_stock_values)

top_10_safety_stock_summary = [summary for summary in filtered_summary.describe_sku(*top_10_safety_stock_skus)]
print(top_10_safety_stock_summary)

