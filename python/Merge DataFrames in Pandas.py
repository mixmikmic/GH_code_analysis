import json
from pandas import DataFrame
import pandas as pd
import numpy as np

EXCEL_SOURCE_FILE = "example_workbook.xlsx"

interface_df = pd.read_excel(EXCEL_SOURCE_FILE, sheetname="interface")
interface_df.head()

port_role_df = pd.read_excel(EXCEL_SOURCE_FILE, sheetname="port_role")
port_role_df.head()

full_intf_df = pd.merge(interface_df, port_role_df, left_on="port_role", right_on="name")
full_intf_df.head(10)

del full_intf_df["name"]
full_intf_df.head(5)

switch_a_intf_df = full_intf_df[full_intf_df.hostname == "Switch_A"]
switch_a_intf_df

del switch_a_intf_df["hostname"]
switch_a_intf_df

# we will use a list comprehension fot this
column_replacements = dict(
    zip(
        switch_a_intf_df.columns,                                         # the current column names
        [e.upper().replace(" ", "_") for e in switch_a_intf_df.columns]   # the new column names
    )
)
switch_a_intf_df = switch_a_intf_df.rename(columns=column_replacements)
switch_a_intf_df

clean_list = switch_a_intf_df.fillna("")
clean_list

result = {
    "HOSTNAME": "Switch A",  # we only have Switch A in this case
    "PORTS": []
}
for index, row in clean_list.iterrows():
    result["PORTS"].append(row.to_dict())

print(json.dumps(result, indent=4))



