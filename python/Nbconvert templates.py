with open("makeitpop.tpl") as f:
    print(f.read())

from IPython.display import HTML
HTML(filename='nbconvert_template_structure.html')

