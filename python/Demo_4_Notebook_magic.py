get_ipython().magic('ls')

get_ipython().magic('pwd')

get_ipython().magic('alias list_files dir')

list_files

get_ipython().run_cell_magic('cmd', '', 'echo This is the command window\ndir *.bat')

get_ipython().run_cell_magic('cmd', '', 'PowerShell Get-ChildItem')

get_ipython().run_cell_magic('script', 'r --no-save', '5 + 5')

get_ipython().system('dir')

mylist = get_ipython().getoutput('dir')
print(mylist)



