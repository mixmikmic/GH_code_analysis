import declarativewidgets as widgets

widgets.init()

get_ipython().run_cell_magic('html', '', '<link rel=\'import\' href=\'urth_components/paper-input/paper-input.html\' \n        is=\'urth-core-import\' package=\'PolymerElements/paper-input\'>\n\n<paper-input label="Enter some text"></paper-input>')

get_ipython().run_cell_magic('html', '', '<link rel="import" href="urth_components/paper-dropdown-menu/paper-dropdown-menu.html" \n        is=\'urth-core-import\' package=\'PolymerElements/paper-dropdown-menu\'>\n<link rel="import" href="urth_components/paper-item/paper-item.html" \n        is=\'urth-core-import\' package=\'PolymerElements/paper-item\'>\n\n<paper-dropdown-menu label="Select Something" noink>\n    <paper-menu class="dropdown-content" attr-for-selected="label">\n        <paper-item label="A">A</paper-item>    \n        <paper-item label="B">B</paper-item>    \n        <paper-item label="C">C</paper-item>    \n    </paper-menu>\n</paper-dropdown-menu>')

