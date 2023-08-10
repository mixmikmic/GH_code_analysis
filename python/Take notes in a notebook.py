get_ipython().system('ls -la')

get_ipython().run_cell_magic('file', 'newfile.txt', '\nWriting some stuff...')

# load file you just made. 
get_ipython().magic('load newfile.txt')

get_ipython().run_cell_magic('file', 'newfile.txt', ' \nOverwriting the text. ')

#take a look at the changed file
get_ipython().magic('load newfile.txt')

get_ipython().run_cell_magic('html', '', '<img src="assets/demo-screenshot.png" width=800 />')



