from arcgis.gis import GIS
gis = GIS("portal url", "username", "password")

me = gis.users.me
me

me.access

import time
# convert Unix epoch time to local time
created_time = time.localtime(me.created/1000)
print("Created: {}/{}/{}".format(created_time[0], created_time[1], created_time[2]))

last_accessed = time.localtime(me.lastLogin/1000)
print("Last active: {}/{}/{}".format(last_accessed[0], last_accessed[1], last_accessed[2]))

print(me.description, " ", me.email, " ", me.firstName, " ", me.lastName, " ", me.fullName)
print(me.level, " ", me.mfaEnabled, " ", me.provider, " ", me.userType)

quota = me.storageQuota
used = me.storageUsage
pc_usage = round((used / quota)*100, 2)
print("Usage: " + str(pc_usage) + "%")

user_groups = me.groups
print("Member of " + str(len(user_groups)) + " groups")

# groups are returned as a dictionary. Lets print the first dict as a sample
user_groups[0]

# anonymous connection to ArcGIS Online
ago_gis = GIS()

# search the users whose email address ends with esri.com
esri_public_accounts = ago_gis.users.search(query='email = @esri.com')
len(esri_public_accounts)

# lets filter out Esri curator accounts from this list
curator_accounts = [acc for acc in esri_public_accounts if acc.username.startswith('Esri_Curator')]
curator_accounts

curator_accounts[0]

esri_hist_maps = ago_gis.users.get(username='Esri_Curator_Historical')
esri_hist_maps

# let us create a built-in account with username: demo_user1 with org_user privilege
demo_user1 = gis.users.create(username = 'demo_user1',
                            password = '0286eb9ac01f',
                            firstname = 'demo',
                            lastname = 'user',
                            email = 'python@esri.com',
                            description = 'Demonstrating how to create users using ArcGIS Python API',
                            role = 'org_user',
                            provider = 'arcgis')

demo_user1

demo_user1_role = demo_user1.role
print(type(demo_user1_role))
print(demo_user1_role)

# create a tiles publisher role
privilege_list = ['portal:publisher:publishTiles',
                 'portal:user:createItem',
                 'portal:user:joinGroup']

tiles_pub_role = gis.users.roles.create(name = 'tiles_publisher',
                                       description = 'User that can publish tile layers',
                                       privileges = privilege_list)

tiles_pub_role

# inspect the privileges of this role
tiles_pub_role.privileges

tiles_pub_user = gis.users.create(username='tiles_publisher',
                                 password = 'b0cb0c9f63e',
                                 firstname = 'tiles',
                                 lastname = 'publisher',
                                 email = 'python@esri.com',
                                 description = 'custom role, can only publish tile layers',
                                 role = 'org_user') #org_user as thats the closest.

tiles_pub_user

tiles_pub_user.privileges

tiles_pub_user.update_role(role = tiles_pub_role)

# query the privileges to confirm
tiles_pub_user.privileges

tiles_pub_user.roleId

searched_role = gis.users.roles.get_role(tiles_pub_user.roleId)
searched_role.description

gis.users.roles.all(max_roles=50)

# let us access an account named publisher1
publisher1 = gis.users.get('publisher1')
publisher1

#list all folders as dictionaries
publisher1_folder_list = publisher1.folders
publisher1_folder_list

# list all items belonging to this user
publisher1_item_list_rootfolder = publisher1.items()
print("Total number of items in root folder: " + str(len(publisher1_item_list_rootfolder)))

#access the first item for a sample
publisher1_item_list_rootfolder[0]

# list all items in the first folder
publisher1.items(folder = publisher1_folder_list[0])

# list the items owned by the user
tiles_pub_user_items = tiles_pub_user.items()
tiles_pub_user_items

# reassign Transport_tiles to publisher1
transport_tiles_item = tiles_pub_user_items[2]
transport_tiles_item

# the reassign_to() method accepts user name as a string. We can also specify a destination folder name
transport_tiles_item.reassign_to(target_owner = 'publisher1', target_folder= 'f1_english')

# now let us get rid of redundant ocean tiles items
tiles_pub_user_items[1].delete()

tiles_pub_user_items[-1].delete()  # an index of -1 in a list refers to the last item

tiles_pub_user.delete(reassign_to='arcgis_python_api')

