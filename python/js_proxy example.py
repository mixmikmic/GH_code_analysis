# First import needed modules and javascript support
# allow import without install
import sys
if ".." not in sys.path:
    sys.path.append("..")

from jp_gene_viz import js_proxy
# this loads the proxy widget javascript "view" implementation
js_proxy.load_javascript_support()
from IPython.display import display

# Then create a "proxy widget" for the jQueryUI dialog.
d = js_proxy.ProxyWidget()

# Construct a command to make the widget into a jQueryUI dialog.
command = d.element().html('<b id="my_dialog">Hello from jQueryUI</b>').dialog()

# Send the command to the widget view (javascript side).
d.send(command)

# Display the widget, which causes the command to also execute.
display(d)

make_visible = d.element().dialog()
d.send(make_visible)

d.results  # The results from the last command are not particularly meaningful.

# We want to put the html from the widget in this list
save_list = []

def save_command_result(result):
    "this is the callback we want to execute when the results arrive"
    #print (result)
    save_list.append(result)

# This "action" gets the html content of the widget.
get_html = d.element().html()

# Send the action to the javascript side for async execution.
d.send(get_html, results_callback=save_command_result)

# If we look at save_list now, it will probably be empty because the
# javascript side has probably not responded yet.
save_list

# But later we should see the HTML saved in the list.
save_list

result = d.evaluate(get_html)
# NOTE: Nothing prints.  I don't know why.
print (result)

print (result)

# get the DOM element associated with the widget from inside the JQuery container.
get_dom_element = d.element().get(0)
dom_element_json = d.evaluate(get_dom_element, level=2)

# Print some info about the JSON for the dom_element sent from Javascript.
print("got " + repr(len(dom_element_json)) + " attributes")
for (i, item) in enumerate(dom_element_json.keys()):
    print(item)
    if i > 10: break
print("...")

# Create the widget.
dp = js_proxy.ProxyWidget()

# Command to populate the widget with an input element with id dp000.
make_input = dp.element().html('<input type="text" id="dp000"/>')._null()

# Command to make the input element into a datepicker and
# fix the style so the datepicker sits on top of the notebook page.
fix_style = (
    dp.window().
    jQuery("#dp000").  # get the jQuery input element by id == "dp".
    datepicker().   # make it a jQuery UI datepicker.
    css("position", "relative").
    css("z-index", "10000").  # put it on top
    attr("size", 55).  # make it big.
    _null()   # we don't care about the command result, discard it.
    )

# Define a python function and data structures to capture
# values sent to the callback when the datepicker input value changes.
identifiers_list = []
arguments_list = []

def dp_change_handler(identifier, arguments):
    "Print the results and also store them in lists."
    print (identifier, arguments['0']['target']['value'])
    identifiers_list.append(identifier)
    arguments_list.append(arguments)
    
# Command to create a "proxy callback" for the change event.
# The proxy will translate values to JSON up to 3 levels deep
# and also send the identifier data "dp has changed" to the handler.
proxy_callback = dp.callback(dp_change_handler, data="dp has changed", level=3)

# Command to associate the proxy callback with the datepicker change event
# using the standard $(x).change(callback) jQuery method.
on_change_command = dp.window().jQuery("#dp000").change(proxy_callback)

# Send the commands to the Javascript view.
dp.send_commands([make_input, fix_style, on_change_command])

# display the widget
display(dp)

document = dp.window().document
new_input = document.createElement("input")
save_input = dp.element()._set("saved_input", new_input)
json_sent = dp.send(save_input)

# what is the type of the new input element?
element_type = dp.evaluate(dp.element().saved_input.type)

# apparently the default type for an input element is "text"
element_type

new = dp.element().New
klass = dp.window().Function
# from https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Function
# emulate "new Function('a', 'b', 'return a + b')"
add_function = new(klass, ["a", "b", "return a + b;"])
save_function = dp.element()._set("my_function", add_function)
json_sent = dp.send(save_function)

function_evaluation = dp.evaluate(dp.element().my_function(34, 6))

function_evaluation

window = dp.window()
ih = dp(window.innerWidth)
dp(window.innerHeight)
dp(ih)
json_sent = dp.flush()

dp.results

new_element_reference = dp.save("another_element", document.createElement("input"))
json_sent = dp.flush()
other_element_type = dp.evaluate(new_element_reference.type)

other_element_type

function_reference = dp.save_new("another_function", klass, ["a", "b", "return a * b;"])
json_sent = dp.flush()
product = dp.evaluate(function_reference(5, 2.2))

product

division = dp.function(["a", "b"], "return a / b;")
tenth = dp.evaluate(division(1.0, 10.0))

tenth

js_div_mod = dp.save_function("div_mod", ["a", "b"], "return {div: Math.trunc(a / b), mod: a % b};")
dp.flush()
d_23_10 = dp.evaluate(js_div_mod(23, 10))

# call the function using the returned reference
d_23_10

# call the function explicitly via the element namespace.
d_467_45 = dp.evaluate(dp.element().div_mod(467, 45))

d_467_45

json_sent = dp.send(dp.function(["element"], "debugger;")(dp.element()))

dp.js_debug()



