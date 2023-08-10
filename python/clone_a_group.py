from arcgis.gis import GIS
from IPython.display import display

# an anonymous connection to ArcGIS Online is sufficient, 
# since we are cloning a public group
source = GIS()

target = GIS("https://python.playground.esri.com/portal", "arcgis_python", "amazing_arcgis_123")

source_group = source.groups.search("title:Vector Basemaps AND owner:esri", outside_org = True)[0]
source_group

source_items = source_group.content()
source_items

import tempfile
if not target.groups.search('Vector Basemaps'):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            thumbnail_file = source_group.download_thumbnail(temp_dir)
            
            #create a group in the target portal with all the properties of the group in the source
            target_group = target.groups.create(title = source_group.title,
                                                 tags = source_group.tags,
                                                 description = source_group.description,
                                                 snippet = source_group.snippet,
                                                 access = source_group.access, 
                                                 thumbnail= thumbnail_file,
                                                 is_invitation_only = True,
                                                 sort_field = 'avgRating',
                                                 sort_order ='asc',
                                                 is_view_only=True)
            #display the group
            display(target_group)
            
    except Exception as e:
        print('Group {} could not be created'.format(source_group.title))
        print(e)
else:
    print('Group {} already exists in the portal'.format(source_group.title))
    target_group = target.groups.search('Vector Basemaps')[0]

#making a list for the items to be cloned in the target portal
items_to_be_cloned = list(source_items)

#checking for the presence of the item in the target portal 
for item in source_items:
    searched_items = target.content.search(query='title:'+item.title, item_type = item.type)   
    
    for s_item in searched_items:
        
        if s_item.title == item.title:
            
            #if an item is not a part of the group in the target portal then share it 
            if s_item not in target_group.content():
                s_item.share(groups= [target_group])
            
            #remove the already existing item from the list of items to be cloned
            items_to_be_cloned.remove(item)                
            
            #display the item
            display(s_item)      
                     
            break

#cloning all items that were not present on the portal before
for item in items_to_be_cloned:    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            thumbnail_file = item.download_thumbnail(temp_dir)
            metadata_file = item.download_metadata(temp_dir)
            target_item_properties = {'title': item.title,
                                      'tags': item.tags,
                                      'text':item.get_data(True),
                                      'type':item.type,
                                      'url':item.url
                                     }       
            #create an item
            target_item = target.content.add(target_item_properties, thumbnail=thumbnail_file)
            
            #share that item with the group on the target portal
            target_item.share(groups=[target_group])
            
            #display the item
            display(target_item)
            
    except Exception as e:
        print('Item {} could not be created in the target portal'.format(item.title))
        print(e)

