import nbformat
nb = nbformat.read('nbconvert.ipynb', as_version=4)

print(nb.cells[2].source)

# Run this, then reload the page to see the change
nb.cells.insert(0, nbformat.v4.new_markdown_cell('**Look at me!**'))
nbformat.write(nb, 'nbconvert.ipynb')

get_ipython().system('jupyter nbconvert --to html nbconvert.ipynb')

get_ipython().system('jupyter nbconvert --to markdown nbconvert.ipynb')

get_ipython().system('cat nbconvert.md')

