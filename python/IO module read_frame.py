import os
import sys
import django
sys.path.append("../")
os.environ['DJANGO_SETTINGS_MODULE'] = 'django_pandas_examples.settings'
django.setup()

from core.models import Employee
from django_pandas.io import read_frame

qs = Employee.objects.all()

df = read_frame(qs)

df

df = read_frame(qs, fieldnames=('age', 'wage', 'full_name'))

df

df = read_frame(qs, fieldnames=('age', 'wage', 'full_name'), index_col='full_name')

df

df = read_frame(qs, fieldnames=('age', 'wage', 'full_name'), index_col='department')

df

