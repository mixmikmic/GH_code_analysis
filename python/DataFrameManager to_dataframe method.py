import os
import sys
import django
sys.path.append("../")
os.environ['DJANGO_SETTINGS_MODULE'] = 'django_pandas_examples.settings'
django.setup()

from core.models import Employee

qs = Employee.objects.all()

df = qs.to_dataframe()

df

df = qs.to_dataframe(fieldnames=['age', 'department', 'wage'])

df

qs.to_dataframe(['age', 'department', 'wage'], index='full_name')

qs.filter(age__gt=20, department='marketing').to_dataframe(index='full_name')

