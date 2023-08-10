from pyDOE import *
import xlwings as xw

parameters = 3
number_of_samples = 1024

wb = xw.Book()
lhd = lhs(parameters, samples=number_of_samples)
wb = xw.Book()
sht = wb.sheets['sheet1']
sht.range('A1').value = lhd



