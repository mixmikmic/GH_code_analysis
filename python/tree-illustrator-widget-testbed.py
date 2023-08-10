get_ipython().run_cell_magic('javascript', '', "// Load JS support for Tree Illustrator widgets \nvar cell = this;\nvar ti, ti2;\n$.getScript(\n    'https://rawgit.com/OpenTreeOfLife/tree-illustrator/master/stylist/ipynb-tree-illustrator.js',\n    function() {\n        // this function will be called once the IPythonTreeIllustrator module has loaded\n        ti = new IPythonTreeIllustrator.IllustratorWidget(cell);\n    }\n);\nalert('hi from %%javascript magic!');")

get_ipython().run_cell_magic('javascript', '', "/* NOTE that calls related to an IllustratorWidget created in a previous cell \n * may fail when a notebook is reloaded. This is because we're outside the \n * safe callback function, and a (re)loading notebook attempts to evaluate \n * all cells in quick succession. If these calls fail with a JS error message\n * below, try re-running these cells by pressing Shift-Click.\n */\nvar ti2 = new IPythonTreeIllustrator.IllustratorWidget(this);  // 'this' is the current cell\n ")

get_ipython().run_cell_magic('javascript', '', 'alert(ti);')



