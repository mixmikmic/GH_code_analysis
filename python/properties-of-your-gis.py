# connect to ArcGIS Online anonymously
from arcgis.gis import GIS
ago_gis = GIS()

ago_gis.properties.portalName

ago_gis.properties.customBaseUrl

org_gis = GIS("https://www.arcgis.com", "username", "password")

org_gis.properties.availableCredits

org_gis.properties.isPortal

org_gis.properties.name

org_gis.properties.maxTokenExpirationMinutes

ent_gis = GIS("portal url", "username", "password")

ent_gis.properties.access

ent_gis.properties.canSignInArcGIS

ent_gis.properties.canSignInIDP

ent_gis.properties.helperServices

ent_gis.properties.isPortal

ent_gis.properties.portalMode

ent_gis.properties.samlEnabled

from IPython.display import display
display(dict(ent_gis.properties))



