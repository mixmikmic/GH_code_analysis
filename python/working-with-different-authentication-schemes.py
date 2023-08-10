# import the GIS class in gis module
from arcgis.gis import GIS

print("ArcGIS Online as anonymous user")    
gis = GIS()
print("Logged in as anonymous user to " + gis.properties.portalName)

gis = GIS("https://python.playground.esri.com/portal")
print("Logged in as anonymous user to " + gis.properties.portalName)

print("ArcGIS Online Org account")    
gis = GIS("https://www.arcgis.com", "arcgis_python", "P@ssword123")
print("Logged in as " + str(gis.properties.user.username))

print("Portal for ArcGIS as a built in user")
gis = GIS("https://portalname.domain.com/webadapter_name", "sharinguser", "password")
print("Logged in as: " + gis.properties.user.username)

print("\n\nBasic Authentication with LDAP")    
ldapbasic = GIS("https://portalname.domain.com/webadapter_name", "amy", "password")
print("Logged in as: " + ldapbasic.properties.user.username)

print("\n\nPortal-tier Authentication with LDAP - enterprise user")
gisldap = GIS("https://portalname.domain.com/webadapter_name", "AVWORLD\\Publisher", "password")
print("Logged in as: " + gisldap.properties.user.username)

print("\n\nPortal-tier Authentication with LDAP - builtin user")    
gisldap = GIS("https://portalname.domain.com/webadapter_name", "sharing1", "password")
print("Logged in as: " + gisldap.properties.user.username)

print("\n\nPKI with key and cert files")  
gis = GIS("https://portalname.domain.com/webcontext", 
          key_file="C:\\path\\to\\key.pem",
          cert_file="C:\\path\\to\\cert.pem")
print("Logged in as: " + gis.properties.user.username)

print("\n\nIntegrated Windows Authentication using NTLM or Kerberos")  
gis = GIS("https://portalname.domain.com/webcontext")
print("Logged in as: " + gis.properties.user.username)

gis = GIS("https://python.playground.esri.com/portal", client_id='f8cRxbP3NO8bf9ag')
print("Successfully logged in as: " + gis.properties.user.username)

gis = GIS("https://python.playground.esri.com/portal", username="arcgis_python", password="amazing_arcgis_123",
          client_id='f8cRxbP3NO8bf9ag')
print("Successfully logged in as: " + gis.properties.user.username)

print("\n\nActive Portal in ArcGIS Pro")  
gis = GIS("pro")

gis = GIS("https://python.playground.esri.com/portal", username='arcgis_python')
print("Successfully logged in as: " + gis.properties.user.username)

#Define 2 different profiles for two different urls
playground_gis = GIS(url="https://python.playground.esri.com/portal", username='arcgis_python', password='amazing_arcgis_123',
                     profile='python_playground_prof')
agol_gis = GIS(url="https://arcgis.com/", username='arcgis_python', password="P@ssword123",
    profile="AGOL_prof")
print("profile defined for {}".format(playground_gis))
print("profile defined for {}".format(agol_gis))

def print_profile_info(gis):
    print("Successfully logged into '{}' via the '{}' user".format(
           gis.properties.portalHostname,
           gis.properties.user.username)) 

playground_gis = GIS(profile='python_playground_prof')
print_profile_info(playground_gis)

agol_gis = GIS(profile="AGOL_prof")
print_profile_info(agol_gis)

