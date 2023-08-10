import os
import sys
import django
sys.path.append("../")
os.environ['DJANGO_SETTINGS_MODULE'] = 'django_pandas_examples.settings'
django.setup()

from core.models import FundVolumeData, Fund, Issuer, Monthly
from django_pandas.io import read_frame
import pandas as pd

pd.options.display.max_rows = 200
pd.options.display.max_columns = 50

read_frame(Issuer.objects.all(), fieldnames=['symbol', 'name'], verbose=False)

funds = read_frame(Fund.objects.all().order_by('issuer__name').filter(active=True), 
             fieldnames=['issuer__name', 'issuer__symbol', 
                         'description', 'symbol'], 
             verbose=False)

funds

qs = FundVolumeData.objects.values_list(
                                    'fund__issuer__symbol', 'fund__symbol', 'period_ending__dateix', 
                                   'total_net_assets_under_management', 'total_unit_holders').filter(
                                    fund__issuer__symbol='UTC',period_ending__dateix__gt='2009-12-31').order_by(
                                    'period_ending__dateix')

fields = [
    'period_ending__dateix', 
    'fund__issuer__symbol', 
    'fund__symbol', 
    'total_net_assets_under_management', 
    'total_unit_holders'
]

pivot_columns = [
    'fund__issuer__symbol', 
    'fund__symbol'
]

ts = qs.to_timeseries(fields,
                     index='period_ending__dateix', pivot_columns=pivot_columns, 
                     values='total_net_assets_under_management',
                     storage='long',
                     verbose=False
                     )

ts

