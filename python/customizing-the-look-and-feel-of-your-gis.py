from arcgis.gis import GIS
gis = GIS("portalname.domain.com/webadaptor", "username", "password")

gis.admin.ux.name

gis.admin.ux.description

gis.admin.ux.description_visibility

gis.admin.ux.name = 'LA PWD GIS'
gis.admin.ux.description = 'Spatial information portal for the Public Works Department of the city of Los Angeles'
gis.admin.ux.description_visibility = True

gis.admin.ux.set_background(background_file='E:\gis_projects\customize_gis\background.png')

gis.admin.ux.set_banner(banner_file = 'E:\gis_projects\customize_gis\banner.png')

gis.admin.ux.set_logo(logo_file='E:\gis_projects\customize_gis\logo.png')

gis.admin.ux.get_banner(download_path = 'E:\gis_projects')

gis.admin.ux.set_background(background_file=None, is_built_in=False)

gis.admin.ux.set_background(is_built_in=True)

#get the list of groups in the GIS and select one of the groups
gis.groups.search()

traffic_group = gis.groups.search()[-1]
traffic_group

gis.admin.ux.featured_content = {'group':traffic_group}

