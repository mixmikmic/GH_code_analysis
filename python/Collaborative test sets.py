import pandas as pd

ratings = pd.read_csv('ratings_updated.csv')
users = pd.read_csv('users.csv')

def get_ratings_subset(block):
    
    block.set_index('userId', inplace=True)
    block = block[['rated_count']]
    
    new_df = ratings.join(block, on='userId', how='left')
    new_df.dropna(inplace=True)
    new_df.drop('rated_count', axis=1, inplace=True)
    new_df.drop('timestamp', axis=1, inplace=True)
    return new_df

users.shape

138493 / 10

users.sort_values('userId', ascending=True, inplace=True)

get_ratings_subset(users.iloc[0:13850]).to_csv('Collab_sets/block1.csv', index=False)

get_ratings_subset(users.iloc[13850:27700]).to_csv('Collab_sets/block2.csv', index=False)



get_ratings_subset(users.iloc[27700:41550]).to_csv('Collab_sets/block3.csv', index=False)

isthisworking = get_ratings_subset(users.iloc[41550:55400])

isthisworking.to_csv('Collab_sets/block4.csv', index=False)

get_ratings_subset(users.iloc[55400:69250]).to_csv('Collab_sets/block5.csv', index=False)

get_ratings_subset(users.iloc[69250:83100]).to_csv('Collab_sets/block6.csv', index=False)

get_ratings_subset(users.iloc[83100:96950]).to_csv('Collab_sets/block7.csv', index=False)

#block 8
get_ratings_subset(users.iloc[96950:110800]).to_csv('Collab_sets/block8.csv', index=False)

#block 9
get_ratings_subset(users.iloc[110800:124650]).to_csv('Collab_sets/block9.csv', index=False)

#block 10
get_ratings_subset(users.iloc[124650:138492]).to_csv('Collab_sets/block10.csv', index=False)

print('done!')

# that didn't work, 100 groups, reload the users csv first
users = pd.read_csv('users.csv')



for i in range(100):
    
    temp_list = []
    for num in range(i, 138493, 100):
        temp_list.append(num)
    
    df = get_ratings_subset(users.loc[temp_list])
    
    filepath = 'Collab_sets/block' + str(i) + '.csv'
    df.to_csv(filepath, index=False)
    
    print(filepath, " ", len(temp_list))
    

print('done')

pd.read_csv('Collab_sets/block2.csv')

pd.read_csv('Collab_sets/block97.csv')



