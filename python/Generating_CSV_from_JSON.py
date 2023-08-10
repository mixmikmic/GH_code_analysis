get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from fastai.conv_learner import *
from fastai.dataset import *

from pathlib import Path
import json
from PIL import ImageDraw, ImageFont
from matplotlib import patches, patheffects

import pandas as pd

PATH = Path('data/pascal')
list(PATH.iterdir())

trn_j = json.load((PATH/'pascal_train2007.json').open())

trn_j.keys()

len(trn_j['images'])
trn_j['images'][:5]

image_df = pd.DataFrame.from_dict(trn_j['images'])
image_df.head()

len(trn_j['annotations'])
trn_j['annotations'][0]

anno_df = pd.DataFrame.from_dict(trn_j['annotations'])
#convert bbox x,y,w,h to y1,x1,y2,x2 (upper left, lower right)
anno_df.bbox = anno_df.bbox.apply(lambda x: [x[1],x[0],x[3]+x[1]-1,x[2]+x[0]-1])
anno_df=anno_df[anno_df.ignore==0].drop('ignore',axis=1)
anno_df.head()
anno_df.shape

cat_df = pd.DataFrame.from_dict(trn_j['categories']).drop('supercategory',axis=1)
cat_df.columns=['category_id','cat_name']
cat_df.head()

cat2id={i['name']:i['id'] for i in trn_j['categories']}
id2cat={i['id']:i['name'] for i in trn_j['categories']}

final_df = anno_df.merge(image_df,how='left',left_on='image_id',
                        right_on='id').drop(['id_x','id_y'],axis=1)

final_df = final_df.merge(cat_df,how='left',on='category_id')

final_df.shape
final_df.head().T

final_df.to_csv(PATH/'all_info.csv',index=False)

# %%time
# #category csv, choose cat with larger bbox area

# def filter_grp(g,col):
#     return g[g[col]==g[col].max()]
# lrg_cat=final_df.groupby(['file_name'],as_index=False).apply(lambda g: filter_grp(g,'area'))
# # lrg_cat.columns=['file_name','cat_name']
# # lrg_cat.head()

final_df[final_df[['file_name','area']].duplicated(keep=False)==True]

#category csv, choose cat with larger bbox area
lrg_cat = final_df.groupby(['file_name'],as_index=False).agg({'area': np.max})
lrg_cat = lrg_cat.merge(final_df[['file_name','cat_name','area']],how='left').drop('area',axis=1)
lrg_cat.head()
lrg_cat.shape

lrg_cat[lrg_cat.duplicated(keep=False)==True] # there might be multiple maximum bboxes' area in 1 pic.
lrg_cat.drop_duplicates('file_name',inplace=True)
# lrg_cat.shape

lrg_cat.shape
# lrg_cat.to_csv(PATH/ 'lrg_cat.csv',index=False)

# bbox csv, choose bbox with larger bbox area
lrg_bbox = final_df.groupby(['file_name'],as_index=False).agg({'area': np.max})
lrg_bbox = lrg_bbox.merge(final_df[['file_name','bbox','area']],how='left').drop('area',axis=1)
lrg_bbox.bbox = lrg_bbox.bbox.apply(lambda x: ' '.join([str(i) for i in x]))
lrg_bbox.drop_duplicates('file_name',inplace=True)
lrg_bbox.head()
lrg_bbox.shape

lrg_bbox.to_csv(PATH/ 'lrg_bbox.csv',index=False)

# multi category csv
def cat_concat(li):
    return ' '.join(str(i) for i in li)
mult_cat = final_df.groupby(['file_name'],as_index=False).agg({'cat_name': cat_concat })
# mult_cat = lrg_cat.merge(final_df[['file_name','cat_name','area']],how='left').drop('area',axis=1)
mult_cat.head()
mult_cat.shape

# multi category csv
def bbox_concat(li):
    return ' '.join(str(i) for l in li for i in l)
mult_bbox = final_df.groupby(['file_name'],as_index=False).agg({'bbox': bbox_concat })
mult_bbox.head()
mult_bbox.shape

mult_cat.to_csv(PATH / 'mult_cat.csv',index=False)
mult_bbox.to_csv(PATH / 'mult_bbox.csv',index=False)

