import json
import requests
import pandas as pd

li = ['Kuen Leung','Jansen Leung','David Wong','Kit Wong Fu','Raymond Choi']

url = "http://www.sfc.hk/publicregWeb/searchByNameJson"
para = {'licstatus':'active',
'searchbyoption':'byname',
'searchlang':'en',
'entityType':'individual',
'searchtext':''}
df_all = pd.DataFrame([])

for name in li[:3]:
    para['searchtext'] = name
    print (name,"....",end='')
    req = requests.post(url,data=para)
    dic = json.loads(req.text)['items']
    if dic:
        df = pd.DataFrame(dic)
        df = df.assign(oname = name)
        df_all = pd.concat([df_all,df])
    print ('ok')

df_all.head()



