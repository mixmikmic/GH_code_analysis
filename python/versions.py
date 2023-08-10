import pandas as pd

print(pd.show_versions())

import datalabframework as dlf

dlf.data.path('.elements.clean.test')

dlf.params.metadata()

dlf.notebook.get_notebook_filename()

dlf.notebook.filename()

dlf.notebook.list_all()

dlf.notebook.statistics('versions.ipynb')

dlf.notebook.execute('main.ipynb')

dlf.project.rootpath()

import main

main.hello()





