import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

df=pd.read_csv('2lev-xc.txt',names=['VRQ','ATTAIN','PID','SEX','SC','SID'],sep='\s',engine='python')
df.head()

print(len(set(df.PID)))
print(len(set(df.SID)))

from hierreg import HierarchicalRegression
from sklearn.model_selection import train_test_split

y=df['ATTAIN']
X=df[['VRQ','SEX','SC']]
groups=df[['PID','SID']]

## simple train-test split - note that I'm suing a random seed to get the same split again later
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
     X, y, groups, test_size=0.33, random_state=42)

## for small problems like this, solver 'ECOS' works faster than the default 'SCS'
## it would still work fine without chaning the default options though, only a bit slower
## note the large regularization parameter
hr=HierarchicalRegression(l2_reg=1e4, cvxpy_opts={'solver':'ECOS'})
hr.fit(X_train, y_train, groups_train)
print(np.mean((y_test-hr.predict(X_test, groups_test))**2))

from sklearn.linear_model import Ridge

y=df['ATTAIN']
X=df[['VRQ','SEX','SC','PID','SID']]
X['PID']=X.PID.map(lambda x: str(x))
X['SID']=X.SID.map(lambda x: str(x))
X=pd.get_dummies(X)

## same train-test split
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

lr=Ridge(100).fit(X_train, y_train)
print(np.mean((y_test-lr.predict(X_test))**2))

rossman=pd.read_csv('train.csv',engine='python')
rossman['StateHoliday']=rossman.StateHoliday.map(lambda x: str(x))

# exlcuding days with no sales
rossman=rossman.loc[rossman.Sales>0]
rossman.head()

rossman.shape

stores=pd.read_csv('C:\\ipython\\mixed effects\\rossman sales\\store.csv')
stores.head()

rossman['Year']=rossman.Date.map(lambda x: x[:4])
rossman['Month']=rossman.Date.map(lambda x: x[5:7])
rossman['DayOfWeek']=rossman.DayOfWeek.map(lambda x: str(x))

max_comp_dist=stores.CompetitionDistance.max()
stores['CompetitionDistance'].loc[stores.CompetitionDistance.isnull()]=max_comp_dist
rossman=pd.merge(rossman,stores[['Store','StoreType','Assortment','CompetitionDistance']],on='Store')
rossman.head()

get_ipython().run_cell_magic('time', '', "from sklearn.linear_model import Ridge\nfrom sklearn.model_selection import train_test_split\nfrom scipy.sparse import coo_matrix, csr_matrix, hstack\n\ny=rossman['Sales']\nX=rossman[['Customers','Open','Promo','StateHoliday','SchoolHoliday','CompetitionDistance']]\nXcateg=rossman[['Store', 'DayOfWeek', 'StoreType','Assortment','Year','Month']]\nXcateg['Store']=Xcateg.Store.map(lambda x: str(x))\nXcateg=coo_matrix(pd.get_dummies(Xcateg).as_matrix())\nX=hstack([pd.get_dummies(X).as_matrix(),Xcateg])\nX_train, X_test, y_train, y_test = train_test_split(\n     X, y, test_size=0.33, random_state=100)\n\nlr=Ridge()\nlr.fit(csr_matrix(X_train),y_train)\npreds_lr=lr.predict(X_test)\nprint(np.sqrt(np.mean((y_test-preds_lr)**2)))")

get_ipython().run_cell_magic('time', '', "from hierreg import HierarchicalRegression\n\ny=rossman['Sales']\nX=rossman[['DayOfWeek','Customers','Open','Promo','StateHoliday','SchoolHoliday','CompetitionDistance','Year','Month']]\ngroup=rossman[['Store','StoreType','Assortment']]\nX_train, X_test, y_train, y_test, group_train, group_test = train_test_split(\n     pd.get_dummies(X), y, group, test_size=0.33, random_state=100)\n\n## for larger datasets, casadi --> IPOPT can provide better running times, but only supports the default parameters\nhlr=HierarchicalRegression(solver_interface='casadi')\nhlr.fit(X_train,y_train,group_train)\npreds_hlr1=hlr.predict(X_test,group_test)\nprint(np.sqrt(np.mean((y_test-preds_hlr1)**2)))")

