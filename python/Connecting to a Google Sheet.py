import pandas as pd

df = pd.read_csv("https://docs.google.com/spreadsheets/d/17Mr201gfDoOTe5ONLS6LYJi1wQbtT26srXeSwUjMK0A/htmlview?usp=sharing&sle=true")

csv_url = "{}/export?gid=0&format=csv"    .format("https://docs.google.com/spreadsheets/d/17Mr201gfDoOTe5ONLS6LYJi1wQbtT26srXeSwUjMK0A")

df = pd.read_csv(csv_url)

df.head()

