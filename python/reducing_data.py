data.to_pickle('pickels/180k_apparel_data')

data=data.loc[~data['formatted_price'].isnull()]
#eleminating data points which do not have price.
#elemination done to reduce the size of data
#reduction in size results increase in speed of processing
#solely done to avoid processing time
#This step can be avoided if waiting time and patience is not taken into account.
print('No. of data points/products after eliminating unpriced products: ',data.shape[0])

#eleminating data points where color is not mentioned.
#elemination done to reduce the size of data
#reduction in size results increase in speed of processing
#solely done to avoid processing time
#This step can be avoided if waiting time and patience is not taken into account.
data=data.loc[~data['color'].isnull()]
print('No. of data points/products after eliminating colorless products: ',data.shape[0])

data.to_pickle('pickels/28k_apparel_data')
#storing data at key stage.

