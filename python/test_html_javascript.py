get_ipython().run_cell_magic('html', '', '<style>\nh2 {\n    color: blue;\n}\n</style>')

get_ipython().run_cell_magic('javascript', '', 'require(["widgets/js/widget", "widgets/js/manager"], function(widget, manager){    \n    console.log(\'IT WORKS! <div />\');\n});')

get_ipython().run_cell_magic('javascript', '', "var cells = document.getElementsByClassName('rendered_html')\ncells[0].style.backgroundColor = 'red';")



