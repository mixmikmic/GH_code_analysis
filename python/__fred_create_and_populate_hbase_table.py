import happybase

# first establish the HBase connection
connection = happybase.Connection('localhost')

# create a table with a column family
def create_and_populate_hbase_table(code_name, csv_file_path):
    if not code_name:
        print "Please provide valid code name!"
        return None
    
    # load CSV file
    try:
        # there are two types of series, one is prefixed with "FRED", another is prefixed
        # with "YAHOO_INDEX", later we need to differentiate them to create different
        # kinds of HBase tables
        series = pd.read_csv(csv_file_path, sep=',', header=0)
    except:
        print "Failed to read csv file!"
        return None
    
    # create HBase table if not exist
    hbase_table_name = "Table_{}".format(code_name)
    try:
        connection.create_table(hbase_table_name, {'cf': {}})
    except Exception as e:
        if "table name already in use" in e.message:
            print "Table name already exists!"
    
    # establish connection to the table
    hbase_table = connection.table(hbase_table_name)

    # populate rows from CSV into HBase table rows
    # https://happybase.readthedocs.io/en/happybase-0.4/tutorial.html
    for i in range(len(series)):
        # need to differentiate code types by prefix
        if code_name.startswith("FRED"): # eg: "FRED_00XAPFEEM086NEST"
            date = series.DATE[i]
            value = series.VALUE[i]
            row_key = "row{}".format(i+1)
            row_content = {'cf:DATE': str(date), 'cf:VALUE': str(value)}
            
        elif code_name.startswith("YAHOO_INDEX"): # eg: "YAHOO_INDEX_GSPC"
            date = series.Date[i]
            adjusted_price = series['Adjusted Close'][i]
            row_key = "row{}".format(i+1)
            row_content = {'cf:DATE': str(date), 'cf:ADJUSTED_PRICE': str(adjusted_price)}
        
        else:
            pass
        
        hbase_table.put(row_key, row_content)

    # count populated row number for sanity check
    total_row = 0
    for key, data in hbase_table.scan():
        total_row += 1
        
    if total_row == len(series):
        print "Successfully create and populate HBase for code {}!".format(code_name)
    else:
        if total_row == 0:
            print "ERROR: Failed to populate HBase table!"
        else:
            print "ERROR: Imcomplete HBase generated!"
    print "Done for this table!"
    
    return hbase_table

# read all downloaded CSV files, create HBase tables for each one
# It took around 0.5 second to parse each CSV file into HBase table, so here
# I only parsed 10 CSV files to demonstrate that it works

import pandas as pd

fred_codes_dir = "/Users/sundeepblue/Desktop/fred_codes"
all_items = os.listdir(fred_codes_dir)
how_many_to_parse = 10
for f in all_items[:how_many_to_parse]:
    if f.startswith("FRED"):
        code_name = f.split(".")[0]
        csv_file_path = os.path.join(fred_codes_dir, f)
        create_and_populate_hbase_table(code_name, csv_file_path)
        

gspc_code_dir = "/Users/sundeepblue/Desktop/gspc_code"
csv_file_path = os.path.join(gspc_code_dir, "YAHOO_INDEX_GSPC.csv")
hbase_table = create_and_populate_hbase_table("YAHOO_INDEX_GSPC", csv_file_path)

# show first 100 rows in the HBase table
rows_to_show = 100
for key, data in hbase_table.scan():
    if rows_to_show >= 0:
        print key, data
    rows_to_show -= 1

# we can order the rows by row_key, but it is unnecessary here

