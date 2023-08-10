import matplotlib.pyplot as plt
import pandas as pd

from supplychainpy.model_inventory import analyse
from supplychainpy.model_demand import simple_exponential_smoothing_forecast
from supplychainpy.sample_data.config import ABS_FILE_PATH
from decimal import Decimal

analyse_kv =dict(
    file_path=ABS_FILE_PATH['COMPLETE_CSV_SM'], 
    start=1, 
    interval_length=12, 
    interval_type='months',
    z_value=Decimal(1.28), 
    reorder_cost=Decimal(400), 
    retail_price=Decimal(455), 
    file_type='csv',
    currency='USD'
)
analysis = analyse(**analyse_kv)
analysis_summary = [ i.orders_summary()for i in analysis]
KR202_209_details = [demand for demand in analysis_summary if demand.get('sku')== 'KR202-209']
print(KR202_209_details[0].get('orders').get('demand'))

ses_df = simple_exponential_smoothing_forecast(demand=KR202_209_details[0].get('orders').get('demand'), length=12, smoothing_level_constant=0.5)
print(ses_df)

FF = ses_df.get('forecast')
regression = ses_df.get('regression')
demand = KR202_209_details[0].get('orders').get('demand')
plt.plot(range(12), demand)
plt.plot(range(12), regression)
plt.show()

from supplychainpy.inventory.summarise import Inventory
filtered_summary = Inventory(analysis)
sku_summary = [summary for summary in filtered_summary.describe_sku('KR202-209')]
print(sku_summary)

