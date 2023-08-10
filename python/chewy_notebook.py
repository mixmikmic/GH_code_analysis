get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# upload chewy data
chewy = pd.read_csv('chewy_final.csv')

pd.set_option('display.max_columns', 50) 
chewy.head(20)



chewy.dtypes

for x in chewy.category.unique():
    print x

print chewy.fish_type.unique()
print chewy.bird_type.unique()
print chewy.small_pet_type.unique()
print chewy.reptile_type.unique()

chewy.breed_size.unique()

len(chewy.item_number.unique())

chewy.cost = chewy.cost.str.replace('$', '')
chewy.cost = chewy.cost.astype(float)

chewy.old_cost = chewy.old_cost.str.replace('$', '')
chewy.old_cost = chewy.old_cost.replace('None', np.nan)
chewy.old_cost = chewy.old_cost.astype(float)

chewy.item_number = chewy.item_number.astype(str)
chewy.page = chewy.page.astype(str)

chewy.describe()

chewy['size'] = chewy['size'].replace('None', np.nan)

chewy['no_reviews'] = chewy['no_reviews'].replace('None', '')

for x in range(len(chewy.weight)):
    if 'pound' in chewy.weight.astype(str)[x]:
        chewy.weight[x] = chewy.weight.astype(str)[x].replace('pounds', '')

for x in range(len(chewy.weight)):
    if 'ounce' in chewy.weight.astype(str)[x]:
        chewy.weight[x] = float(chewy.weight.astype(str)[x].replace('ounces', ''))/16.0

chewy.weight = chewy.weight.astype(float)

chewy.drop('dimensions', 1, inplace = True)

chewy['sale'] = (chewy.old_cost-chewy.cost)/chewy.old_cost
chewy['sale'] = chewy.sale*100
chewy.sale = chewy.sale.apply(lambda x: round(x,2))
chewy.sale = chewy.sale.astype(float)

chewy['no_reviews'] = chewy['no_reviews'].replace('', np.nan )
chewy.no_reviews = chewy.no_reviews.astype(float)

chewy['pet'] = chewy['category'].apply(lambda x: x.split(' ')).apply(lambda x: x[0])
chewy.pet = chewy.pet.replace("Small", "Small Pet")
chewy.pet = chewy.pet.replace("Gifts", np.nan)


chewy.pet.unique()

chewy.percent_rec = chewy.percent_rec.astype(float)

chewy['item_type'] = chewy['category'].apply(lambda x: x.split(' ')).apply(lambda x: x[1:])
chewy['item_type'] = chewy['item_type'].apply(lambda x: ' '.join(x))

chewy.item_type.unique()

for x in chewy.item_type.unique():
    print x

chewy.item_type.replace('& Books', 'Gifts & Books', inplace = True)
chewy.item_type.replace('Pet Food & Treats', 'Food & Treats', inplace = True)
chewy.item_type.replace('Pet Supplies', 'Supplies', inplace = True)
chewy.item_type.replace('Pet Beds, Hideouts & Toys', 'Beds, Hiedouts & Toys', inplace = True)

chewy.item_type.replace('Treats', 'Food & Treats', inplace = True)
chewy.item_type.replace('Food', 'Food & Treats', inplace = True)

chewy.item_type.replace('Training & Behavior', 'Training & Cleaning', inplace = True)
chewy.item_type.replace('Cleaning & Potty', 'Training & Cleaning', inplace = True)

sorted(chewy.brand.unique())

cols = ['item_number', 'pet', 'item_type',
       'category',
        'page',
        'brand',
        'product_name',
        'product_description',
        'no_reviews',
        'rating',
        'percent_rec',
        'cost',
        'old_cost',
        'sale',
        'food_form',
        'food_texture',
        'special_diet',
        'supplement_form',
        'weight',
        'size',
        'lifestage',
        'breed_size',
        'fish_type',
        'bird_type',
        'reptile_type',
        'small_pet_type',
        'made_in',
        'material',
       'toy_feature',
       'bowl_feature',
       'leash_and_collar_feature',
       'litter_box_type',
       'litter_and_bedding_type',
       'bed_type']
chewy = chewy[cols]

chewy = chewy.sort_values('item_number')
chewy

chewy.to_csv('chewy_update.csv', index = False)

chewy = pd.read_csv('chewy_update.csv')

chewy.category.value_counts()

chewy.item_type.value_counts()

plt.figure(figsize = (12,6))
plt.hist(chewy.rating, bins = 20, color = "#6767ff")

plt.hist(chewy.percent_rec, bins = 25)

plt.hist(chewy.cost, bins = 100)
plt.axis([0,200,0,4000])

plt.hist(chewy['old_cost'].dropna(), bins = 100)
plt.axis([0,200,0,2200])

plt.hist(chewy['sale'].dropna(), bins = 50)
plt.axis([0,100,0,900])

plt.hist(chewy['weight'].dropna())
# plt.axis([0,200,0,2200])

plt.hist(chewy['no_reviews'].dropna(), bins = 100)
plt.axis([0,500,0,4000])

plt.figure(figsize=(12, 6))
sns.jointplot(chewy.rating, chewy.cost)

sns.jointplot(chewy.no_reviews, chewy.cost)

sns.jointplot(chewy.rating, chewy.sale, size = 8,)

plt.figure(figsize = (12,10))
sns.boxplot(x='pet', y='cost', data=chewy).set(ylim = (0,100))

sns.boxplot(x='brand', y='cost', data=chewy)

sns.boxplot(x='category', y='sale', data=chewy)

sns.boxplot(x='category', y='sale', data=chewy)

sns.lmplot("weight", "cost", chewy)

sns.lmplot("rating", "cost", chewy)

sns.lmplot("percent_rec", "cost", chewy)

sns.lmplot("old_cost", "cost", chewy)

sns.lmplot("old_cost", "cost", chewy, hue = "pet")

sns.lmplot("weight", "cost", chewy)

PetGrid = sns.FacetGrid(chewy, col='pet', hue="pet", palette="Set1", size=3, col_wrap = 4)
PetGrid.map(sns.distplot, "cost").set(xlim=(0,100)).add_legend() 

tipGrid = sns.PairGrid(chewy)
tipGrid.map(plt.scatter)

# supposed to add , scatter_kws={'alpha':0.3} to args to get transparency

chewy_subset = chewy

chewy.groupby('pet').agg(['mean', 'min', 'max'])

chewy.groupby('pet').plot.scatter('cost', 'no_reviews')

chewy.groupby('pet')['cost'].median().sort_values(ascending = False).plot.bar()

chewy.groupby('pet')['cost'].mean().sort_values(ascending = False).plot.bar()

# from scipy.stats import mode
# chewy.groupby('pet')['cost'].mode().sort_values(ascending = False).plot.bar()

chewy.groupby('item_type')['sale'].median().sort_values().plot.bar()

chewy.groupby('item_type')['cost'].median().sort_values().plot.bar()

chewy.groupby('item_type')['old_cost'].median().sort_values().plot.bar()

chewy.groupby('item_type')['old_cost'].mean().sort_values().plot.bar()

plt.figure(figsize = (12,10))
chewy.groupby('item_type')['cost'].mean().sort_values().plot.barh(color = 'orange')
plt.ylabel('Item Category')
plt.xlabel('Average Cost')

chewy.groupby('item_type')['rating'].mean().sort_values(ascending = False).round(2)

chewy.groupby('pet')['sale'].mean().sort_values().plot.barh()

chewy_by_pet = chewy.groupby('pet')

sns.boxplot(x = 'pet', y = 'sale', data = chewy)

chewy_subset = chewy.ix[:,['pet', 'sale', 'cost', 'old_cost', 'no_reviews', 'rating', 'percent_rec', 'weight']]
g = sns.PairGrid(chewy_subset.dropna(), hue = 'pet', palette='Set2')

g.map_diag(plt.hist)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
g.add_legend()

for key, values in chewy_by_pet:    #because its a tuple, pull out both elements
    print key
    print '-'*55
    print values
    print '\n'

i = 0
for item in chewy_by_pet:
    print item[0] + ' : ' + str(len(item[1]))
    i += len(item[1])
print 'total pets: ' + str(i)

i = 0
for item in chewy.groupby('brand'):
    print item[0] + ' : ' + str(len(item[1]))
    i += len(item[1])
print 'total pets: ' + str(i)

chewy_by_pet[['cost', 'sale', 'percent_rec']].agg(['mean', 'median']).round(1)

chewy.groupby(['pet', 'item_type'])['cost'].agg(['mean', 'count'])

chewy.groupby(['brand', 'item_type'])['cost'].mean().sort_values()

chewy.groupby(['brand', 'item_type'])['rating'].mean().sort_values()





k = chewy.groupby('brand')['no_reviews'].count()
k.to_csv('k.csv')

import plotly.plotly as py
import plotly.graph_objs as go

trace = go.Heatmap(z=chewy.cost,
                   x=chewy.pet,
                   y=chewy.item_type)
data=[trace]
py.iplot(data, filename='labelled-heatmap')

chewy.groupby('brand')['no_reviews'].count().sort_values(ascending=False)

dogfood = chewy.loc[chewy.item_type=='Food & Treats',:].loc[chewy.pet == 'Dog']
dogfood_by_brand =dogfood.groupby('brand')[['cost', 'sale', 'percent_rec', 'rating', 'no_reviews']].mean().round(2)
dogfood_by_brand = dogfood_by_brand.loc[dogfood_by_brand.no_reviews>45]
dogfood_by_brand.sort_values('no_reviews', ascending = False)



sns.lmplot("percent_rec", "cost", dogfood_by_brand)

sns.lmplot("percent_rec", "sale", dogfood_by_brand)

dogfood = chewy.loc[chewy.item_type=='Food & Treats',:].loc[chewy.pet == 'Dog']
print dogfood.loc[dogfood.special_diet.notnull()][['cost', 'rating', 'percent_rec']].mean().round(2)
print dogfood.loc[dogfood.special_diet.isnull()][['cost', 'rating', 'percent_rec']].mean().round(2)

