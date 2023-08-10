import pandas as pd

# Import the excel file and call it xls_file
xls_file = pd.Excelfile('Data/exmaple.xls')
xls_flle

# View the excel file's sheet names
xls_file.sheet_names

# Load the xls file's Sheet1 as a dataframe
df = xls_file.parse('Sheet1')
df

