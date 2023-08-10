import os
import os.path as osp
import json
from datetime import datetime
from pprint import pprint
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10, 10)

annotations_dir = '/playpen/data/vist/annotations'
images_dir = '/playpen/data/vist/images'

split = 'test'
# Load dii's split json
b = datetime.now()
path_to_dii_val = osp.join(annotations_dir, 'dii', split+'.description-in-isolation.json')
dii = json.load(open(path_to_dii_val))
print 'dii\'s [%s] loaded. It took %.2f seconds.' % (split, (datetime.now() - b).total_seconds())

# Load sis's split json
b = datetime.now()
path_to_sis_val = osp.join(annotations_dir, 'sis', split+'.story-in-sequence.json')
sis = json.load(open(path_to_sis_val))
print 'sis\'s [%s] loaded. It took %.2f seconds.' % (split, (datetime.now() - b).total_seconds())

# Let's check one ann
sis.keys()

sis['annotations'][0]

sis['images'][0]

sis['albums'][3]

def show_album(alb_id):
    img_ids = alb_to_img_ids[alb_id]
    plt.figure()
    cols = 5
    rows = math.ceil(len(img_ids)/float(cols))
    for i, img_id in enumerate(img_ids):
        img = Images[img_id]
        img_file = osp.join(images_dir, split, img['id']+'.jpg')
        img_content = imread(img_file)
        img_content = imresize(img_content, (224, 224))
        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(img_content)
        ax.axis('off')
#         ax.set_title(str(img_id))
        ax.set_title(str(img_id)+'\n'+img['datetaken'][4:])
    plt.show()

def show_story(story_id, show_image=True):
    sent_ids = story_to_sent_ids[story_id]
    if show_image:
        plt.figure()
        for i, sent_id in enumerate(sent_ids):
            img_id = Sents[sent_id]['img_id']
            img = Images[img_id]
            img_file = osp.join(images_dir, split, str(img_id)+'.jpg')
            img_content = imread(img_file)
            img_content = imresize(img_content, (224, 224))
            ax = plt.subplot(1, len(sent_ids), i+1)
            ax.imshow(img_content)
            ax.axis('off')
            ax.set_title(str(img_id)+'\n'+img['datetaken'][5:])
        plt.show()
    for sent_id in sent_ids:
        sent = Sents[sent_id]
        print 'order[%s], image_id[%s], text[%s]' % (sent['order'], sent['img_id'], sent['original_text'])

Images = {item['id']: item for item in sis['images']}
Albums = {item['id']: item for item in sis['albums']}
alb_to_img_ids = {}
for item in sis['images']:
    alb_id = item['album_id']
    img_id = item['id']
    alb_to_img_ids[alb_id] = alb_to_img_ids.get(alb_id, []) + [img_id]

# sort img_ids based on datetime
def getDateTime(img_id):
    x = Images[img_id]['datetaken']
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
for alb_id, img_ids in alb_to_img_ids.items():
    img_ids.sort(key=getDateTime)

# make sents = [{}]
sents = []
for ann in sis['annotations']:
    sent = ann[0].copy()
    sent['id'] = sent.pop('storylet_id')
    sent['order'] = sent.pop('worker_arranged_photo_order')
    sent['img_id'] = sent.pop('photo_flickr_id')
    sents += [sent]
Sents = {sent['id']: sent for sent in sents}

# story_id -> sent_ids
story_to_sent_ids = {}
for sent_id, sent in Sents.items():
    story_id = sent['story_id']
    story_to_sent_ids[story_id] = story_to_sent_ids.get(story_id, []) + [sent_id]

def get_order(sent_id):
    return Sents[sent_id]['order']
for story_id, sent_ids in story_to_sent_ids.items():
    sent_ids.sort(key=get_order)
    
# alb_id -> story_ids
alb_to_story_ids = {}
for story_id, sent_ids in story_to_sent_ids.items():
    sent = Sents[sent_ids[0]]
    alb_id = sent['album_id']
    alb_to_story_ids[alb_id] = alb_to_story_ids.get(alb_id, []) + [story_id]

alb_id = '72157607155047588'
show_album(alb_id)

story_ids = alb_to_story_ids[alb_id]
print 'This album has %s stories.' % len(story_ids)
show_story(story_ids[4], True)

# alb_ids = Albums.keys()
# alb_id = alb_ids[2]; print alb_id
# show_album(alb_id)

sis['type']

dii.keys()

pprint(dii['annotations'][0][0])
pprint(dii['annotations'][5][0])

sis['annotations'][0][0]

sis['albums'][0]['id']

for i in range(len(sis['albums'])):
    if not sis['albums'][i]['id'] == dii['albums'][i]['id']:
        print 'inconsitancy found.'

sis['albums'][0]

sis['images'][7]['id']

sis['annotations'][14]

dii['annotations'][14]

for i in range(len(sis['annotations'][:2])):
    sd = sis['annotations'][i][0]
    dd = dii['annotations'][i][0]
    if sd['album_id'] != dd['album_id'] or sd['photo_flickr_id'] != dd['photo_flickr_id']          or sd['worker_arranged_photo_order'] != dd['photo_order_in_story']:
            print 'k'



