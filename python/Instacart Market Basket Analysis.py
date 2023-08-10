import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import gensim
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

get_ipython().run_line_magic('matplotlib', 'inline')

# Make plots larger
plt.rcParams['figure.figsize'] = (15, 9)

#Loading all the csv files

df_aisles = pd.read_csv("aisles.csv")
df_departments = pd.read_csv("departments.csv")
df_order_products_prior = pd.read_csv("order_products__prior.csv")
df_order_products_train = pd.read_csv("order_products__train.csv")
df_orders = pd.read_csv("orders.csv")
df_products = pd.read_csv("products.csv")

#Reading the orders.csv file
df_orders.head()

#Counting the number of rows and columns in orders.csv
df_orders.shape

#Finding if the dataset has any null values
df_orders.isnull().sum()

#Count no. of rows in each dataset
cnt_srs = df_orders.eval_set.value_counts()
plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Eval set type', fontsize=12)
plt.title('Count of rows in each dataset', fontsize=15)
plt.show()
print(cnt_srs)

#Finding number of customers
def get_unique_count(x):             ## Defining a function to get unique count for user_id from orders.csv
    return len(np.unique(x))

cnt_srs = df_orders.groupby("eval_set")["user_id"].aggregate(get_unique_count)
cnt_srs

#Validating prior order range
cnt_srs = df_orders.groupby("user_id")["order_number"].aggregate(np.max).reset_index()
cnt_srs = cnt_srs.order_number.value_counts()

#Bar-graph for the order-reorder counts
plt.figure(figsize=(20,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color = 'red')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Maximum order number', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

#Changing the data labels into name of days of weeks
import calendar
days=[]

for i in df_orders['order_dow']:
    days.append(calendar.day_name[i])

#Adding another column for day-name of the week as per the number
df_orders['converted_dow']=days

#Finding out the busiest day of the week
plt.figure(figsize=(12,8))
sns.countplot(x="converted_dow", data=df_orders, order=df_orders['converted_dow'].value_counts().index, color='teal')
plt.ylabel('Count', fontsize=12)
plt.xlabel('Day of week', fontsize=12)
plt.title("Frequency of order by week day", fontsize=15)
plt.show()
cnt_dow = df_orders.groupby('order_dow')['order_id'].aggregate(get_unique_count)
cnt_dow

#Figuring out which time of the day is the busiest
plt.figure(figsize=(12,8))
sns.countplot(x="order_hour_of_day", data=df_orders, color='teal')
plt.ylabel('Count', fontsize=12)
plt.xlabel('Hour of day', fontsize=12)
plt.title("Frequency of order by hour of the day", fontsize=15)
plt.show()
cnt_hod = df_orders.groupby('order_hour_of_day')['order_id'].aggregate(get_unique_count)
cnt_hod

#Heat-map for the intersection of the day of week and hour of day
grouped_df = df_orders.groupby(["order_dow", "order_hour_of_day"])["order_number"].aggregate("count").reset_index()
grouped_df = grouped_df.pivot('order_dow', 'order_hour_of_day', 'order_number')

plt.figure(figsize=(12,6))
sns.heatmap(grouped_df)
plt.title("Frequency of Day of week Vs Hour of day")
plt.show()

#Finding out after how many days an order is reordered
plt.figure(figsize=(20,8))
sns.countplot(x="days_since_prior_order", data=df_orders, color='grey')
plt.ylabel('Count', fontsize=12)
plt.xlabel('Days since prior order', fontsize=12)
plt.title("Frequency distribution by days since prior order", fontsize=15)
plt.show()

# percentage of re-orders in prior set #
df_order_products_prior.reordered.sum() / df_order_products_prior.shape[0]

# percentage of re-orders in train set #
df_order_products_train.reordered.sum() / df_order_products_train.shape[0]

#No. of products per order on an average
grouped_df = df_order_products_train.groupby("order_id")["add_to_cart_order"].aggregate("max").reset_index()
cnt_srs = grouped_df.add_to_cart_order.value_counts()

plt.figure(figsize=(20,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color = 'teal')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Number of products in the given order', fontsize=12)
#plt.xticks(rotation='vertical')
plt.show()

df_order_products_prior = pd.merge(df_order_products_prior, df_products, on='product_id', how='left')
df_order_products_prior = pd.merge(df_order_products_prior, df_aisles, on='aisle_id', how='left')
df_order_products_prior = pd.merge(df_order_products_prior, df_departments, on='department_id', how='left')
df_order_products_prior.head()

#Counting the total no of products purchased, i.e. the most popular products.
cnt_srs = df_order_products_prior['product_name'].value_counts().reset_index().head(20)
cnt_srs.columns = ['product_name', 'frequency_count']
cnt_srs

#Count for the sales according to the aisles
cnt_srs = df_order_products_prior['aisle'].value_counts().head(20)
plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='teal')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Aisle', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

# Pie-chart for department-wise sales.
plt.figure(figsize=(10,10))
temp_series = df_order_products_prior['department'].value_counts().head(5)
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
plt.pie(sizes, labels=labels, 
        autopct='%1.1f%%', startangle=200)
plt.title("Departments distribution", fontsize=15)
plt.show()

# Finding the reorder ratio with respect to Department.
# This means if a certain product is ordered previously, what are the chances that it will be reordered again.
grouped_df = df_order_products_prior.groupby(["department"])["reordered"].aggregate("mean").reset_index()
plt.figure(figsize=(20,9))
sns.pointplot(grouped_df['department'].values, grouped_df['reordered'].values, alpha=0.8, color='teal')
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Department', fontsize=12)
plt.title("Department wise reorder ratio", fontsize=15)
plt.xticks(rotation='vertical',fontsize=14)
plt.show()

#Add to cart order - reorder ratio
# Here we are trying to understand if the order in which a product was added to the cart will affect it's chances of reordering
df_order_products_prior["add_to_cart_order_mod"] = df_order_products_prior["add_to_cart_order"].copy() # Making a copy of order_products_prior
df_order_products_prior["add_to_cart_order_mod"].loc[df_order_products_prior["add_to_cart_order_mod"]>30] = 30
grouped_df = df_order_products_prior.groupby(["add_to_cart_order_mod"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_df['add_to_cart_order_mod'].values, grouped_df['reordered'].values, alpha=0.8, color='teal')
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Add to cart order', fontsize=12)
plt.title("Add to cart order - Reorder ratio", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()

# Reorder ratio across the day of week
df_order_products_train = pd.merge(df_order_products_train, df_orders, on='order_id', how='left')
grouped_df = df_order_products_train.groupby(["order_dow"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.barplot(grouped_df['order_dow'].values, grouped_df['reordered'].values, alpha=0.8, color='teal')
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Day of week', fontsize=12)
plt.title("Reorder ratio across day of week", fontsize=15)
plt.xticks(rotation='vertical')
plt.ylim(0.5, 0.7)
plt.show()

#Reorder ration across hour od day
grouped_df = df_order_products_train.groupby(["order_hour_of_day"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.barplot(grouped_df['order_hour_of_day'].values, grouped_df['reordered'].values, alpha=0.8, color='teal')
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Hour of day', fontsize=12)
plt.title("Reorder ratio across hour of day", fontsize=15)
plt.xticks(rotation='vertical')
plt.ylim(0.5, 0.7)
plt.show()

#Reorder ration across hour od day
grouped_df = df_order_products_train.groupby(["order_hour_of_day"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.barplot(grouped_df['order_hour_of_day'].values, grouped_df['reordered'].values, alpha=0.8, color='teal')
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Hour of day', fontsize=12)
plt.title("Reorder ratio across hour of day", fontsize=15)
plt.xticks(rotation='vertical')
plt.ylim(0.5, 0.7)
plt.show()

# Reading csv's
train_orders = pd.read_csv("order_products__train.csv")
prior_orders = pd.read_csv("order_products__prior.csv")
products = pd.read_csv("products.csv").set_index('product_id')

# Coverting product_id's to string
train_orders["product_id"] = train_orders["product_id"].astype(str)
prior_orders["product_id"] = prior_orders["product_id"].astype(str)

# Grouping by order_id
train_products = train_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())
prior_products = prior_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())

# Joining 
sentences = prior_products.append(train_products)
longest = np.max(sentences.apply(len))
sentences = sentences.values

# Using the Word2Vec package from Gensim model
model = gensim.models.Word2Vec(sentences, size=100, window=longest, min_count=2, workers=4)

# Creating a list
vocab = list(model.wv.vocab.keys())

# Importing PCS from Ski-kit learn
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(model.wv.syn0)

def get_batch(vocab, model, n_batches=3):
    output = list()
    for i in range(0, n_batches):
        rand_int = np.random.randint(len(vocab), size=1)[0]
        suggestions = model.most_similar(positive=[vocab[rand_int]], topn=5)
        suggest = list()
        for i in suggestions:
            suggest.append(i[0])
        output += suggest
        output.append(vocab[rand_int])
    return output
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(21, 21))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

embeds = []
labels = []
for item in get_batch(vocab, model, n_batches=4):
    embeds.append(model[item])
    labels.append(products.loc[int(item)]['product_name'])
embeds = np.array(embeds)
embeds = pca.fit_transform(embeds)
plot_with_labels(embeds, labels)

# Created a bag of words to find out which cultures product are selling the most.
df_bow = pd.read_csv("bow.csv")

#Bag of Words with Name of Cultures
df_bow.Culture_names.head(5)

for x in df_bow.Culture_names :
    print(x)

df_products.product_name

#Dictionary
cultname=[]

#Splitting each Product in sepaarte Words
for i in df_products.product_name:
    a=i.split()
    for j in df_bow.Culture_names:
        if j in a:
            cultname.append(j)
            

#Combining the pair in dictionary
counts = dict()
for i in cultname:
    counts[i] = counts.get(i,0) + 1
print(counts)

counts.keys()

#Bar plot of Dictionary
plt.figure(figsize=(18,10))
plt.bar(counts.keys(),counts.values(),color='b')
plt.xticks(rotation='vertical')
plt.show()

