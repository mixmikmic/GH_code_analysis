from arcgis.gis import GIS

#connect to host GIS, in this case, this is an ArcGIS Enterprise instance
host_gis = GIS("https://host-portal.company.com/portal", "username")

#connect to guest GIS, in this case this is an org on ArcGIS Online
guest_gis = GIS("https://guest-portal.company.com/portal", "username")

#search for the traffic analysis group in host gis
host_group = host_gis.groups.search("Traffic")[0]
host_group

#create a collaboration
description='Data sharing initiative between Dept.' +             'of Tranportation and Police of the city of Philadelphia'
host_collab = host_gis.admin.collaborations.create(name='Philly Police Dept. + Dept. of Transport', 
                                                   description=description,
                                                   workspace_name='Philly Police Transport data sharing',
                                                   workspace_description=description,
                                                   portal_group_id=host_group.id,
                                                   host_contact_first_name='Traffic',
                                                   host_contact_last_name='Chief',
                                                   host_contact_email_address='traffic.chief@city.gov',
                                                  access_mode='sendAndReceive')

host_collab

#get the list of workspaces available as part of the collaboration
host_collab.workspaces

#compose the list of collaboration workspaces and the privileges for each
config = [{host_collab.workspaces[0]['id']:'sendAndReceive'}]

#invite the guest GIS and download the invitation file
invite_file = host_collab.invite_participant(config_json= config, expiration=24,
                                             guest_portal_url = "https://guest-portal.company.com/portal",
                                            save_path=r'E:\gis_projects\collab')

#print the path to the invite file
invite_file

guest_gis.admin.collaborations.accept_invitation(first_name='Police', 
                                                 last_name='Chief', 
                                                 email='police.chief@city.gov', 
                                                 invitation_file=invite_file)

#get the list of collaborations on the guest GIS and pick the one created earlier
guest_gis.admin.collaborations.list()

#in this case, there are two collaborations and the second is the relevant one
guest_collab = guest_gis.admin.collaborations.list()[1]
type(guest_collab)

response_file = guest_collab.export_invitation(out_folder = r"E:\gis_projects\collab")
response_file

#first get the list of worksapces in the guest collaboration
guest_collab.workspaces

#search for the crime analysis group in the guest portal
guest_group = guest_gis.groups.search("crime")[0]
guest_group

guest_collab.add_group_to_workspace(portal_group = guest_group, 
                                    workspace= guest_collab.workspaces[0])

host_collab.import_invitation_response(response_file)

host_gis.admin.collaborations.collaborate_with(guest_gis = guest_gis,
                                               collaboration_name='Transport_PD_data_sharing', 
                                               collaboration_description='Data sharing initiative between' + \
                                               'the transport and Police departments')

print("Collaborations on host GIS")
for collab in host_gis.admin.collaborations.list():
    print(collab.name)

print("-----------------------------")
print("Collaborations on guest GIS")
for collab in guest_gis.admin.collaborations.list():
    print(collab.name)

