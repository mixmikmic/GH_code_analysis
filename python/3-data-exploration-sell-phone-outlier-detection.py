import pandas as pd

get_ipython().magic('pylab inline')

df = pd.read_excel("../data/sell_your_phone.xlsx")

df.head()

df.describe()

df = df.dropna()

df = df.drop(['Timestamp'],axis=1)

#Some default stuff for my plotting
aspect_mult = 0.9
figsize(aspect_mult*16,aspect_mult*9)
linewidth = 3

#One Hot Encoding
cat_cols = ["Brand","Broken_Screen"]
df_continuous = pd.get_dummies(df,columns=cat_cols)
df_continuous.columns

df_continuous.head()

plt.scatter(df_continuous['Internal Storage Size in Gigabytes (GB)'], 
            df_continuous.Buying_Price, 
            s = 200)
#Add some context to the plot
plt.title("Scatter: Storage Size (GB) vs Original Buying Price")
plt.xlabel("Storage Size (GB)")
plt.ylabel("Original Buying Price")
plt.grid()

from sklearn.cluster import KMeans

X = df_continuous.values
reps = 3

within_cluster = []
x_range = range(2,31)
for k in x_range:
    temp_wc = []
    for i in range(reps):
        clf = KMeans(n_clusters=k,n_jobs=-1)
        clf.fit(X)
        temp_wc.append(clf.inertia_)
    within_cluster.append(temp_wc)
within_cluster = np.array(within_cluster)

pyplot.errorbar(x_range,np.mean(within_cluster,axis=1),
                yerr = np.std(within_cluster,axis=1), linewidth=linewidth)
plt.ylabel("Within cluster sume of squares")
plt.xlabel("$k$")
plt.grid()

k = 10

clf = KMeans(n_clusters=k,n_jobs=-1)
clf.fit(X)
cluster_labels = clf.predict(X)

c= ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, k)]
cluster_colors = []

plt.scatter(df_continuous['Internal Storage Size in Gigabytes (GB)'],
            df_continuous.Buying_Price, c=cluster_labels,
            s=30,label=cluster_labels)
plt.title("Clustering Bag Words $k=20$",fontsize=20)
plt.title("K-Means Clustering $k$=" + str(k)+ " Scatter: Storage Size (GB) vs Original Buying Price")
plt.xlabel("Storage Size (GB)")
plt.ylabel("Original Buying Price")
plt.grid()

X_standardised = (df_continuous.values -np.mean(df_continuous.values))/np.std(df_continuous.values)

