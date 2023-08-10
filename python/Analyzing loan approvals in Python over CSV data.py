from io import StringIO

import requests
import json
import pandas as pd
import brunel

df = pd.read_csv("https://raw.githubusercontent.com/ODMDev/decisions-on-spark/master/data/miniloan/miniloan-decisions-ls-10K.csv")
df.head()

total_rows = df.shape[0]
print("The size of the decision set is " + str(total_rows))

get_ipython().run_line_magic('brunel', 'data(\'df\') stack polar bar x("const") y(#count) color(approval) legends(none) label(approval) :: width=200, height=300')

get_ipython().run_line_magic('brunel', "data('df') chord x(approval) y(creditScore) color(#count) tooltip(#all)")

get_ipython().run_line_magic('brunel', "data('df') x(income) y(creditScore) color(approval:yellow-green) :: width=800, height=300")

get_ipython().run_line_magic('brunel', "data('df') x(loanAmount) y(creditScore) color(approval:yellow-green) :: width=800, height=300")

get_ipython().run_line_magic('brunel', 'data(\'df\') bar x(loanAmount) y(#count) bin(loanAmount) style("size:100%") :: width=800, height=300')

