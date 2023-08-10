import nbformat
nb = nbformat.read('Notebook file format.ipynb', as_version=4)

print(nb.cells[2].source)

# Run this, then reload the page to see the change
nb.cells.append(nbformat.v4.new_markdown_cell('**Look at me!**'))
nbformat.write(nb, 'Notebook file format.ipynb')

