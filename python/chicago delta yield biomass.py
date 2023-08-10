import psycopg2
import pandas as pd
pgconn = psycopg2.connect(database='sustainablecorn', host='iemdb', user='nobody')
cursor = pgconn.cursor()
pd.set_printoptions(max_rows=400, max_columns=10)
cursor.execute("""
 SELECT a.site, a.plotid, varname, year, value, p.rotation, p.tillage from agronomic_data a JOIN plotids p
 ON (p.uniqueid = a.site and p.plotid = a.plotid) where
 varname in ('AGR7', 'AGR17', 'AGR19', 'AGR39') and 
 value ~* '[0-9\.]' and value != '.' and value !~* '<' 
""")
data = {}
for row in cursor:
    key = '%s|%s|%s' % (row[0], row[1], row[3])
    if not data.has_key(key):
        data[key] = {'rotation': row[5], 'tillage': row[6]}
    data[key][ row[2] ] = float(row[4])

rows = []
for key in data.keys():
    tokens = key.split("|")
    rows.append( dict(siteid=tokens[0], plotid=row[1], year=tokens[2], 
                      agr7=data[key].get('AGR7'), 
                      agr17=data[key].get('AGR17'), 
                      agr19=data[key].get('AGR19'), 
                      agr39=data[key].get('AGR39'),
                      rotation=data[key]['rotation'],
                      tillage=data[key]['tillage']
                      ) )
    
df = pd.DataFrame(rows)
df.fillna(np.nan)
print 'Loaded %s rows from the database!' % (len(df),)

def a(ar):
    if len(ar[ar.notnull()]) == 0:
        return None
    return np.ma.average(ar[ar.notnull()])

rows = []
for year in ["2011", "2012", "2013"]:
    for sid in df.siteid.unique():
        agr7 = df[(df.siteid==sid)&(df.year==year)].agr7
        if len(agr7[agr7.notnull()]) == 0:
            continue
        agr7 = np.average(agr7[agr7.notnull()])
        
        cyield_nocc = a( df[(df.siteid==sid)&(df.year==year)&
                       ((df.rotation=='ROT4')|(df.rotation=='ROT5'))].agr17)
        syield_nocc = a( df[(df.siteid==sid)&(df.year==year)&
                       ((df.rotation=='ROT4')|(df.rotation=='ROT5'))].agr19)
        cyield_cc = a( df[(df.siteid==sid)&(df.year==year)&
                       ((df.rotation=='ROT36')|(df.rotation=='ROT37'))].agr17)
        syield_cc = a( df[(df.siteid==sid)&(df.year==year)&
                       ((df.rotation=='ROT36')|(df.rotation=='ROT37'))].agr19)

        rows.append( dict(siteid=sid, year=year, ryebio=agr7, 
                          cyield_nocc=cyield_nocc, syield_nocc=syield_nocc, 
                          cyield_cc=cyield_cc,     syield_cc=syield_cc) )
        
df2 = pd.DataFrame(rows)

df2.sort('ryebio')

df3 = df2[(df2.siteid!='WOOSTER.COV')]
(fig, ax) = plt.subplots(1,1)

ax.scatter(df3.ryebio, df3.cyield_cc - df3.cyield_nocc, marker='+', s=50, label='Corn')
ax.scatter(df3.ryebio, df3.syield_cc - df3.syield_nocc, marker='s', s=50, label='Soy')
ax.legend()
ax.set_ylabel("$\Delta$ Yield (CoverCrop minus Non) [kg/ha]")
ax.set_xlabel("Rye Spring Biomass [kg/ha]")
ax.set_title("Change in Yield with Spring Biomass")
ax.grid(True)



