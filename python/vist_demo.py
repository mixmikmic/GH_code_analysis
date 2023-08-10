import vist
import matplotlib.pyplot as plt
from datetime import datetime
from pprint import pprint
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10, 10)

vist_images_dir = '/playpen/data/vist/images'
vist_annotations_dir = '/playpen/data/vist/annotations'
sis = vist.Story_in_Sequence(vist_images_dir, vist_annotations_dir)

album_id = sis.Albums.keys()[0]
sis.show_album(album_id)
album = sis.Albums[album_id]
# pprint(album)

story_ids = sis.Albums[album_id]['story_ids']
story_id = story_ids[0]
sis.show_story(story_id)
print sis.Stories[story_id]['img_ids']

# albums stats
split_to_album_ids = {'train': [], 'val': [], 'test': []}
total_albums = 0
for split in ['train', 'val', 'test']:
    split_to_album_ids[split] = [album_id for album_id, album in sis.Albums.items() if album['split'] == split]
    print 'There are [%s] albums in [%s] split.' % (len(split_to_album_ids[split]), split)
    total_albums += len(split_to_album_ids[split])
print 'In total, there are [%s] albums.' % total_albums

# stories stats
split_to_story_ids = {'train': [], 'val': [], 'test': []}
for story in sis.stories:
    album_id = story['album_id']
    split = sis.Albums[album_id]['split']
    split_to_story_ids[split] += [story['id']]
total_stories = 0
for split in ['train', 'val', 'test']:
    print 'There are [%s] stories in [%s] split.' % (len(split_to_story_ids[split]), split)
    total_stories += len(split_to_story_ids[split])
print 'In total, there are [%s] stories.' % (total_stories)

# sents stats
print 'SIS:'
split_to_sent_ids = {'train': [], 'val': [], 'test': []}
for sent in sis.sents:
    album_id = sent['album_id']
    split = sis.Albums[album_id]['split']
    split_to_sent_ids[split] += [sent['id']]
total_sents = 0
for split in ['train', 'val', 'test']:
    print 'There are [%s] sents in [%s] split.' % (len(split_to_sent_ids[split]), split)
    total_sents += len(split_to_sent_ids[split])
print 'In total, there are [%s] sents.' % (total_sents)

# check story order
def check_dts_order(dts):
    flag = True
    for i in range(1, len(dts)):
        if dts[i] <= dts[i-1]:
            flag = False
    return flag

inorder = 0
for story in sis.stories:
    dts = []
    for i, sent_id in enumerate(story['sent_ids']):
        sent = sis.Sents[sent_id]
        assert sent['order'] == i
        img = sis.Images[sent['img_id']]
        dt = datetime.strptime(img['datetaken'], '%Y-%m-%d %H:%M:%S')
        dts += [dt]
    if check_dts_order(dts):
        inorder += 1
print 'Among %s stories, %s [%.2f%%] are in order' % (len(sis.stories), inorder, inorder*100.0/len(sis.stories))

# img_ids = sis.Stories[story_id]['img_ids']
# dii.show_imgs_with_sents(img_ids, False)

# Load DII instance
dii = vist.Description_in_Isolation(vist_images_dir, vist_annotations_dir)

# DII's sents stats
print 'DII:'
dii_split_to_sent_ids = {'train': [], 'val': [], 'test': []}
for sent in dii.sents:
    album_id = sent['album_id']
    split = dii.Albums[album_id]['split']
    dii_split_to_sent_ids[split] += [sent['id']]
total_sents = 0
for split in ['train', 'val', 'test']:
    print 'There are [%s] sents in [%s] split.' % (len(dii_split_to_sent_ids[split]), split)
    total_sents += len(dii_split_to_sent_ids[split])
print 'In total, there are [%s] sents.' % (total_sents)



