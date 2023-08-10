import json
from arcgis.gis import Item

def update_webmaps(gis, id_mappings, items=None, org=False):
    """
    Allows for the find/replace of item ids and urls in a webmap service.

    This method allows for the quick updating a collection of WebMap Items.

    =====================     ====================================================================
    **Argument**              **Description**
    ---------------------     --------------------------------------------------------------------
    id_mappings               Required dictionary.  A mapping set of the old value to the new value.

                              Example:

                              {
                                  'abcd123456' : 'efgh7890123',
                                  'http://myoldservice.esri.com/spam', 'http://mynewservice.esri.com/eggs'
                              }
    ---------------------     --------------------------------------------------------------------
    items                     Optional list. List of web map items to update. If no item list if
                              provided, the current user item's will be search and update all WebMap
                              Items.
    ---------------------     --------------------------------------------------------------------
    org                       optional boolean.  If True, all WebMap will be updated.  If False,
                              webmap the current user owns will be updated. This parameter is only
                              valid if the items parameter is None.
    =====================     ====================================================================

    :returns: boolean

    .. note::
    On failure of an item, the value will be False, and the items that could not be updated will be returned.

    """
    
    cm = gis.content
    if items is None and org == False:
        items = cm.search(
            query="owner: %s" % dict(gis.properties['user'])['username'],
            item_type="Web Map",
            max_items=10000)
    elif org == True and items is None:
        items = cm.search(query="*",
                            item_type="Web Map",
                            max_items=10000)
    results = {'success': True, 'notUpdated' : [] }
    for idx, item in enumerate(items):
        if isinstance(item, str):
            item = cm.get(item)
        if isinstance(item, Item) and            item.type == 'Web Map':
            data = json.dumps(item.get_data())
            for k,v in id_mappings.items():
                data = data.replace(k,v)
            res = item.update(data=data)
            if res == False:
                results['notUpdated'].append(item)
        else:
            results['notUpdated'].append(item)
        del idx, item
    return results

from arcgis.gis import GIS
gis = GIS(url="https://mysite.supercool.com/portal", username="*****", password='****************')
print(update_webmaps(gis=gis, id_mappings={'39e499c73d8c418c9b049463cf572327' : '7711af7cc8ad4644ba5d997df4c87e18'}))

items = gis.content.search(query="owner: %s" % dict(gis.properties['user'])['username'],
                  item_type="Web Map",
                  max_items=10000)
for item in items: # Verify all the WebMaps have been updated.
    print(all([(layer['itemId'] == '39e499c73d8c418c9b049463cf572327') == False            for layer in item.get_data()['operationalLayers']]))

