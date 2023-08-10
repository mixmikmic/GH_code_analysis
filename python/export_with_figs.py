import pardir; pardir.pardir() # Allow imports from parent directory
import nbformat
nb_raw = ""
with open("tmp_fibonaccistretch_executed.nbconvert.ipynb", "r") as f:
    nb_raw = f.read()
nb = nbformat.reads(nb_raw, as_version=4)

str(nb.cells[6])[:1000]
print(str(nb.cells[12])[:2000])
# nb.cells[12]

# from traitlets.config import Config

# # 1. Import the exporter
# from nbconvert import HTMLExporter

# # 2. Instantiate the exporter. We use the `basic` template for now; we'll get into more details
# # later about how to customize the exporter further.
# html_exporter = HTMLExporter()
# html_exporter.template_file = 'basic'

# # 3. Process the notebook we loaded earlier
# (body, resources) = html_exporter.from_notebook_node(jake_notebook)

# create a configuration object that changes the preprocessors
from traitlets.config import Config
from nbconvert import HTMLExporter

c = Config()
c.HTMLExporter.preprocessors = ['nbconvert.preprocessors.ExtractOutputPreprocessor']

# create the new exporter using the custom config
html_exporter_with_figs = HTMLExporter(config=c)
html_exporter_with_figs.preprocessors

(body, resources) = html_exporter_with_figs.from_notebook_node(nb)

print("\nresources with extracted figures (notice that there's one more field called 'outputs'):")
print(sorted(resources.keys()))

print("\nthe actual figures are:")
print(sorted(resources['outputs'].keys()))

with open("../fibonaccistretch_with_figs.html", "w") as f:
    f.write(body)

resources["outputs"][sorted(resources["outputs"].keys())[0]][:100]

import os

figs_dir = "../data/figs"
for k in resources["outputs"].keys():
    v = resources["outputs"][k]
    path = os.path.join(figs_dir, k)
    print(path)
    with open(path, "wb") as f:
        f.write(v)



