import os
import pandas as pd

train = pd.read_csv('train.csv', index_col='Image')

whaleIDs = list(train['whaleID'].unique())

for w in whaleIDs:
    os.makedirs('./imgs/'+w)

for image in train.index:
    folder = train.loc[image, 'whaleID']
    old = './imgs/{}'.format(image)
    new = './imgs/{}/{}'.format(folder, image)
    try:
        os.rename(old, new)
    except:
        print('{} - {}'.format(image,folder))

get_ipython().system('ls -1 ./imgs/*/*.jpg | wc -l')
get_ipython().system('grep jpg train.csv | wc -l')



