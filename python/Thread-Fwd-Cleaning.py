import pandas as pd

keiko = pd.read_csv('../data/keiko_categorized.csv')
rohan = pd.read_csv('../data/rohan_labeled.csv')
joyce = pd.read_csv('../data/joyce_labeled.csv')
fatou = pd.read_csv('../data/fatou_relabeled.csv')
kristian = pd.read_csv('../data/kristian_labeled.csv')

joyce = pd.read_csv('../data/joyce_labeled.csv')

count2 = fatou.Body

import re

expr2 = "on [.]{3} [.]{1,2}, [.]{4}, [.]{1,2}:[.]{1,2} [.]{2}.*"

expr1 = "on [.]{3} [.]{1,2}, [.]{4}.*"

expr3 = "on .{3}, .{3} .{1,2}, .{4}, .{1,2}:.{1,2}.*"

expr4 = "on .{6,9}, .{3,9} .{1,2}, .{4}.*"

expr5 = "on .{3} .{1,2}, .{4}.*"

expr6 = "on .{3}, .{3} .{1,2}, .{4}.*"

expr7 = ".{3}年.{1,2}月.{1,2}日 下..{1,2}:.{1,2}.*"

expr8 = "begin forwarded me|begin forwarded message|sent with mailtrack"

expr9 = ".*--- forwarded message?"

expr10 = ".*--- forwarded message .*subject:"

expr11 = ".. .. ..., .... .{1,2}, .{4} . ... .{1,2}:.{1,2}.*"

expr12 = "on .{1,2} .{3} .{4}, at .{1,2}:.{1,2}.*"

expression = expr1+"|"+expr2+"|"+expr3+"|"+expr4+"|"+expr5+"|"+expr6+"|"+expr7+"|"+expr8+"|"+expr10+"|"+expr9+"|"+expr11+"|"+expr12

expression

def remove_xtra(content):
    return re.sub(expression, '', content)

joyce['Body'] = joyce['Body'].fillna('')

joyce.Body = kristian.Body.apply(remove_xtra)

content = kristian['Body']
#for i, em in enumerate(content):
    #print('EMAIL!!! ', i+1, '\n','  ', em, '\n')

len(fatou.Body)

kristian.Body = kristian.Body.dropna()

len(kristian.Body)

keiko.to_csv('../data/keiko_categorized.csv', index=False, sep = ',', encoding='utf-8')

rohan.to_csv('../data/rohan_labeled.csv', index=False, sep = ',', encoding='utf-8')

joyce.to_csv('../data/joyce_labeled.csv', index=False, sep = ',', encoding='utf-8')

fatou.to_csv('../data/fatou_relabeled.csv', index=False, sep = ',', encoding='utf-8')

kristian.to_csv('../data/kristian_labeled.csv',  index=False, sep = ',', encoding='utf-8')

import pandas as pd

keiko = pd.read_csv('../data/keiko_categorized.csv')
rohan = pd.read_csv('../data/rohan_labeled.csv')
joyce = pd.read_csv('../data/joyce_labeled.csv')
fatou = pd.read_csv('../data/fatou_relabeled.csv')
kristian = pd.read_csv('../data/kristian_labeled.csv')

#keiko.head()

get_ipython().run_line_magic('pinfo', 'pd.concat')

dfs = [rohan, keiko, fatou, kristian, joyce]

together = pd.concat(dfs)

#together.to_csv('../data/recombined.csv', index=False, sep = ',', encoding='utf-8')



