# weatherman.py
# 主程式
# P.S.這裡抓不到，因為沒有準備實體函式在資料中

import report  #把 report.py 在相同目錄資料夾中，即可對其呼叫匯入
description = report.get_description()  #使用report中的get_description函數

from get_description import report as get

print("Today's weather:", description)

# report.py
# 函式庫

def get_description():
    """Return random weather, just like the pros"""
    from random import choice   #匯入標準函式庫 random 中的 choice函數
    possibilities = ['rain', 'snow', 'sleet', 'fog', 'sun', 'who knows']
    return choice(possibilities)

