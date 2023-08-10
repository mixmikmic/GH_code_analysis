# First import needed modules and javascript support
# allow import without install
import sys
if ".." not in sys.path:
    sys.path.append("..")
    
# Load needed libraries
from jp_gene_viz import js_proxy
# this loads the proxy widget javascript "view" implementation
js_proxy.load_javascript_support()
from jp_gene_viz import js_context
import ipywidgets as widgets
from IPython.display import display

js_context.load_if_not_loaded(["simple_upload_button.js"])
# To support binary data the internal unicode string passes
# hexidecimal encoded bytes by default.
from jp_gene_viz.file_chooser_widget import from_hex

u = js_proxy.ProxyWidget()
output = widgets.Textarea(description="output")

def upload_handler(dummy, args):
    "Do something with the uploaded metadata and content."
    info = args["0"]
    # decode hexidecimal content
    content = from_hex(info["hexcontent"])
    L = [info["name"], info["type"], str(info["size"]), "===", content]
    output.value = "\n".join(L)

# Need the callback to provide 2 levels when passing arguments back.
upload_callback = u.callback(upload_handler, data=None, level=2)

div = u.element()
u(div.simple_upload_button(upload_callback).appendTo(div))
u.flush()
v = widgets.VBox(children=[u, output])
display(v)

from jp_gene_viz.file_chooser_widget import FileChooser
c = FileChooser(root=".", upload=True, message="upload or download")
c.enable_downloads()
c.show()



