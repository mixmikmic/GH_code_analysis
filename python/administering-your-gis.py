from arcgis.gis import GIS
gis = GIS("https://portalname.domain.com/webadaptor", "username", "password")

gis.admin.license.all()

pro_license = gis.admin.license.get('ArcGIS Pro')
pro_license

type(pro_license)

#get all users licensed for ArcGIS Pro
pro_license.all()

get_ipython().run_line_magic('matplotlib', 'inline')
pro_license.plot()

pro_license.report

pro_license.user_entitlement('username')

pro_license.assign(username='arcgis_python', entitlements='desktopBasicN')

pro_license.revoke(username='arcgis_python', entitlements='*')

gis.admin.credits.credits

gis.admin.credits.enable()

gis.admin.credits.is_enabled

gis.admin.credits.default_limit

#assign one thenth of the available licenses to arcgis_python account
api_acc_credits = gis.admin.credits.credits / 10
gis.admin.credits.allocate(username='arcgis_python', credits=api_acc_credits)

api_acc = gis.users.get('arcgis_python')
api_acc

api_acc.assignedCredits

api_acc.availableCredits

rohit = gis.users.get('rsingh_geosaurus')
rohit.availableCredits

gis.admin.credits.disable()

gis.admin.federation.servers

gis.admin.federation.validate_all()

gis.admin.federation.servers[1]['id']

gis.admin.federation.unfederate('GFyaVzJXiogsxKxH')

gis.admin.logs.settings

import datetime
import pandas as pd
now = datetime.datetime.now()
start_time = now - datetime.timedelta(days=10)
start_time

recent_logs = gis.admin.logs.query(start_time = start_time)

#print a message as a sample
recent_logs['logMessages'][0]

log_df = pd.DataFrame.from_records(recent_logs)
log_df.head(5) #display the first 5 records

log_df.to_csv('./portal_logs_last_10_days.csv')

gis.admin.logs.clean()

existing_policy = gis.admin.password_policy.policy
existing_policy

from copy import deepcopy
new_policy = deepcopy(existing_policy)
new_policy['passwordPolicy']['minLength'] = 10
new_policy['passwordPolicy']['minUpper'] = 1
new_policy['passwordPolicy']['minLower'] = 1
new_policy['passwordPolicy']['minDigit'] = 1
new_policy['passwordPolicy']['minOther'] = 1
new_policy['passwordPolicy']['expirationInDays'] = 90
new_policy['passwordPolicy']['historySize'] = 5

gis.admin.password_policy.policy = new_policy['passwordPolicy']

gis.admin.password_policy.policy

gis.admin.password_policy.reset()

gis.admin.security.config

gis.admin.security.ssl.list()

portal_cert = gis.admin.security.ssl.list()[0]
portal_cert.export(out_path = './')

gis.admin.security.enterpriseusers

gis.admin.security.groups.properties

gis.admin.system.licenses.properties

from datetime import datetime
datetime.fromtimestamp(round(gis.admin.system.licenses.properties.expiration/1000))

gis.admin.system.licenses.release_license('username')

gis.admin.machines.list()

mac1 = gis.admin.machines.list()[0]
mac1.properties

mac1.status()

portal_dir_list = gis.admin.system.directories
portal_dir_list[0].properties

for portal_dir in portal_dir_list:
    print(portal_dir.properties.name + " | " + portal_dir.properties.physicalPath)

gis.admin.system.web_adaptors.list()

wa = gis.admin.system.web_adaptors.list()[0]
wa.properties

wa.url

gis.admin.system.database

gis.admin.system.index_status

gis.admin.system.languages

