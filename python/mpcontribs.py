# not during workshop
get_ipython().system('pmg config --add PMG_MAPI_KEY <YOUR_API_KEY>')
get_ipython().system('cd ~/work/MPContribs && git pull && git submodule update webtzite && git submodule update mpcontribs/users')

from mpcontribs.users.mp_workshop_2017.rest.rester import MpWorkshop2017Rester

mpr = MpWorkshop2017Rester(test_site=True)

mpr.preamble

[doc['_id'] for doc in mpr.query_contributions()]

mpfile = mpr.find_contribution('598a3342a25ec601ef334003')
mpid = mpfile.ids[0]
mpid

mpfile.hdata[mpid]['data'] # dictionary

mpfile.gdata[mpid]

mpr.get_contributions() # DataFrame
# also mpr.get_graphs() to get all graphs for workshop

