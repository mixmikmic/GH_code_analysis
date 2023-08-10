get_ipython().run_cell_magic('sh', '', 'python wctool.py -h')

get_ipython().run_cell_magic('sh', '', 'python wctool.py -l')

get_ipython().run_cell_magic('sh', '', 'python wctool.py -g -id "93fac4d7-2fab-4a02-b282-ce28a1d9f3f5"')

get_ipython().run_cell_magic('sh', '', 'python wctool.py -g -id "93fac4d7-2fab-4a02-b282-ce28a1d9f3f5" -full')

get_ipython().run_cell_magic('sh', '', 'python wctool.py -g -id "93fac4d7-2fab-4a02-b282-ce28a1d9f3f5" -o henriksWorkspace.json')

get_ipython().run_cell_magic('sh', '', 'python wctool.py -c -name HenriksNewWorkspace -desc "Test for my blog" -lang de -i henriksWorkspace.json')

get_ipython().run_cell_magic('sh', '', 'python wctool.py -u -id "f5db2447-55dd-4c7e-a764-2319a17dbfee" -name "Henrik renamed it"')

get_ipython().run_cell_magic('sh', '', 'python wctool.py -u -id "f5db2447-55dd-4c7e-a764-2319a17dbfee" -intents -counterexamples -i henriksWorkspace.json')

get_ipython().run_cell_magic('sh', '', 'python wctool.py -d -id "f5db2447-55dd-4c7e-a764-2319a17dbfee"')

get_ipython().run_cell_magic('sh', '', 'python wctool.py -logs -id "09969794-a510-4eab-95f3-b482d94a7ac6" -filter "request.input.text:Hello"')

get_ipython().run_cell_magic('sh', '', 'python wctool.py -dialog -id "09969794-a510-4eab-95f3-b482d94a7ac6"')



