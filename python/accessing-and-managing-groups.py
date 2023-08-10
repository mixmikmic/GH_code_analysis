from arcgis.gis import GIS
ago_gis = GIS()
urban_groups = ago_gis.groups.search('title:urban', max_groups=15)
urban_groups

urban_groups[3]

esri_owned_groups = ago_gis.groups.search(query='owner:esri and description:basemaps', max_groups=15)
esri_owned_groups

antartic_basemaps = esri_owned_groups[0]
antartic_basemaps.access

import time
print(antartic_basemaps.groupid, antartic_basemaps.isFav, antartic_basemaps.isInvitationOnly)
print(antartic_basemaps.owner)
time.localtime(antartic_basemaps.created/1000)

ago_gis.groups.get(antartic_basemaps.groupid)

# connect to GIS with credentials
gis = GIS("https://portal url", "username", "password")
geocaching_group = gis.groups.create(title='Recreational geocaching',
                                    tags = 'hobby, geocaching, gps, hide n seek',
                                    description = 'Group to share your landmarks and games',
                                    snippet = 'Share your GPX tracks as feature layers here',
                                    access = 'org',
                                    is_invitation_only = 'False',
                                    thumbnail = r'D:\temp\geocaching.jpg')
geocaching_group

forest_falls_game = gis.content.get('252c3a4d2c64428c9ffccffe2ae0ff1e')
forest_falls_game.access

# this item is private, let us share it to the group so other enthusiasts can enjoy this map
forest_falls_game.share(groups=geocaching_group.id)

geocaching_group.content()

# let us add publisher1 and publisher2 to this group
geocaching_group.add_users(['publisher1', 'publisher2', 'demo_user1'])

# remove the demo_user1 account wrongly added to this group
geocaching_group.remove_users(['demo_user1'])

# can you remove the owner of the group?
geocaching_group.remove_users(['arcgis_python_api'])

geocaching_group.get_members()

geocaching_group.update(is_invitation_only=True)

geocaching_group.isInvitationOnly

# let us reassign ownership to publisher1
geocaching_group.reassign_to(target_owner = 'publisher1')

# let the logged in user leave the group
geocaching_group.leave()

# now query the group's members to confirm the new onwer and member list
geocaching_group.get_members()

geocaching_group.content()

geocaching_group.delete()

