from arcgis.gis import GIS

from partnerutils.clone_utils import search_group_title

gis = GIS(username="mpayson_startups")

# Group Schema
# users_update_items must be false to invite members from other organizations
group_schema = {
    "title": "My Test Title",
    "tags": "test, group, poc, scripts",
    "description": "Test group for partner python scripts",
    "access": 'private',
    "is_invitation_only": True,
    "users_update_items": False
}

# create the group if it doesn't already exist
group = search_group_title(gis, group_schema["title"])
if group is None:
    group = gis.groups.create_from_dict(group_schema)
group

usernames = ['mspatialstartups']

group.invite_users(usernames)

