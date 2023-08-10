pwd

get_ipython().run_line_magic('run', '../__init__.py')
get_ipython().run_line_magic('matplotlib', 'inline')

target = pd.read_csv('../Datasets/madelon_train.labels', sep=' ', header=None)
train = pd.read_csv('../Datasets/madelon_train.data', sep=' ', header=None)
val = pd.read_csv('../Datasets/madelon_valid.data', sep=' ', header=None)

train.head()

target.columns = ['target']
train = train.drop(train.columns[500], axis=1)

train= pd.concat([train, target], 1)

X = train.drop(['target'], axis=1)
y = train['target']

train.head()

X.shape, y.shape

X.head()

X.describe()

plt.scatter(X[3],X[:4],c=y)

# #correlation matrix
plt.figure(figsize=(10,7), dpi=80)
plt.title('UCI_EDA')
plt.pcolor(np.corrcoef(X, rowvar=False),
           cmap=mpl.cm.magma, 
           alpha=0.8)
plt.colorbar()
plt.savefig('../Images/uci_heatmap.png')
plt.show()

pca = PCA(2)
X_pc = pca.fit_transform(X)

fig = plt.figure(figsize=(12,10))
fig.add_subplot(2,4,1)
plt.scatter(X[0],X[1], c=y)
fig.add_subplot(2,4,2)
plt.scatter(X[0],X[2], c=y)
fig.add_subplot(2,4,3)
plt.scatter(X[0],X[3], c=y)
fig.add_subplot(2,4,4)
plt.scatter(X[0],X[4], c=y)
fig.add_subplot(2,4,5)
plt.scatter(X[1],X[2], c=y)
fig.add_subplot(2,4,6)
plt.scatter(X[1],X[3], c=y)
fig.add_subplot(2,4,7)
plt.scatter(X[2],X[3], c=y)
fig.add_subplot(2,4,8)
plt.scatter(X_pc[:,0],X_pc[:,1], c=y)

database_1 = pd.read_pickle('../Datasets/database_1.p')
database_2 = pd.read_pickle('../Datasets/database_2.p')
database_3 = pd.read_pickle('../Datasets/database_3.p')

database_1.head()

database_1.describe()

# database_1.isnull().sum().sort_values(ascending=False)

# database_2.isnull().sum().sort_values(ascending=False)

# database_3.isnull().sum().sort_values(ascending=False)

sample = database_1.sample(200)

plt.figure(figsize=(10,7), dpi=80)
plt.title('Database_EDA')
plt.pcolor(np.corrcoef(sample, rowvar=False),
           cmap=mpl.cm.magma, 
           alpha=0.8)
plt.colorbar()
plt.savefig('../Images/db_heatmap.png')
plt.show()



