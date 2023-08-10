print(data['product_type_name'].describe())
'''For 183,138 products we have 72 unique product type names.
Out of 72 product names, "SHIRT" is most frequent
frequency(SHIRT) = 167,794
% (SHIRT)=(167794/183138)*100=91.62% '''

#name of unique/different product types
print(data["product_type_name"].unique())

#find the 10 most frequent product type names
product_type_count = Counter(list(data['product_type_name']))
print(product_type_count.most_common(10))

print(data['brand'].describe())
'''183,138-182,987=151 missing values
Out of 10577 unique values most frequent is "Zago"
freq(Zago)=223
%(Zago)=(223/182987)*100 = 0.121 % '''

brand_count=Counter(list(data['brand']))
brand_count.most_common(10)

print(data['color'].describe())

#7380 unique colors
#missing colors=183138-64956=118182
#7.2% products are black
#35.4% of products have color information

color_count=Counter(list(data['color']))
color_count.most_common(10)

data['formatted_price'].describe()

#Only 15.5% of products with price information

price_count=Counter(list(data['formatted_price']))
price_count.most_common(10)

print(data['title'].describe())

data.to_pickle('pickels/180k_apparel_data')



