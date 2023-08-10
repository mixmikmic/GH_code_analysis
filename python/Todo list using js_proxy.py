# First import needed modules and javascript support
# allow local import without install
import sys
if ".." not in sys.path:
    sys.path.append("..")

from jp_gene_viz import js_proxy
# this loads the proxy widget javascript "view" implementation
js_proxy.load_javascript_support()
from IPython.display import display
import ipywidgets as widgets

class ToDoApp:
    
    """
    Framework for an IPython notebook todo list which allows
    additions and deletions to a sorted list of things to do.
    """
    
    def __init__(self):
        "build the widgets and assemble them into a display."
        self.description_to_time = {}
        hd = self.hour_dropdown = widgets.Dropdown(
            options=map(str, range(24)), width="20px")
        hd.layout.width = "70px"
        md = self.minute_dropdown = widgets.Dropdown(
            options=map(str, range(0, 60, 10)), width="20px")
        md.layout.width = "70px"
        dp = self.date_picker = self.make_date_picker()
        desc = self.description_text = widgets.Text()
        dialog = self.dialog = self.make_modal_dialog()
        b = self.add_button = widgets.Button(description="add")
        b.on_click(self.add_click)
        td = self.todo_list_area = self.make_todo_list_area()
        vih = self.vertical_input("hour", hd)
        vim = self.vertical_input("minute", md)
        vid = self.vertical_input("date", dp)
        videsc = self.vertical_input("description", desc)
        top = widgets.HBox(children=[vih, vim, vid, videsc, b])
        self.assembly = widgets.VBox(children=[top, td, dialog])
        
    def add_click(self, b):
        "add a todo entry in response to a click on the 'add' button."
        hour = int(self.hour_dropdown.value)
        minute = int(self.minute_dropdown.value)
        date = self.get_date()
        try:
            [dd, mm, yy] = map(int, date.split("/"))
        except Exception:
            return self.warning("expect date format dd/mm/yyyy")
        description = self.description_text.value.strip()
        if not description:
            return self.warning("description must not be empty")
        timestamp = (yy, mm, dd, hour, minute)
        self.description_to_time[description] = timestamp
        self.format_todo_list()
        
    def format_todo_list(self):
        "Format the current list of todo items in order by timestamp."
        d2t = self.description_to_time
        list_area = self.todo_list_area
        element = list_area.element()
        jQuery = list_area.window().jQuery
        list_area(element.empty())
        if not d2t:
            list_area(element.html("No todo entries remain."))
        else:
            L = sorted((d2t[d], d) for d in d2t)
            for (t, description) in L:
                (yy, mm, dd, hour, minute) = t
                desc_text = "<span>%s:%s %s/%s/%s %s &nbsp;</span>" % (hour, minute, dd, mm, yy, description)
                delete_callback = list_area.callback(self.delete_todo, data=description)
                button = jQuery("<button>delete</button>").click(delete_callback)
                div = jQuery("<div/>").append(desc_text).append(button)
                list_area(element.append(div))
        list_area.flush()
        
    def delete_todo(self, description, js_arguments):
        "handle a 'delete' callback: delete the item with the matching descriptoin and re-format."
        del self.description_to_time[description]
        self.format_todo_list()
        
    def vertical_input(self, label, widget):
        "label and input widget with label above and widget below."
        h = widgets.HTML(label)
        return widgets.VBox(children=[h, widget])
        
    def show(self):
        "Display the todo list mini-application."
        display(self.assembly)
        
    def make_todo_list_area(self):
        "Make a js_proxy widget to contain the todo list."
        list_area = js_proxy.ProxyWidget()
        element = list_area.element()
        list_area(element.html("No todos yet."))
        list_area.flush()
        return list_area
        
    def make_date_picker(self):
        "Make a js_proxy widget containing a date picker."
        picker = js_proxy.ProxyWidget()
        element = picker.element()
        # use jQuery to a datepicker to the picker widget
        # which sits on top of the notebook, in the right place.
        jQuery = picker.window().jQuery
        datepicker_input = (
            jQuery('<input type="text" size=20/>').
            datepicker().
            css("position", "relative").
            css("z-index", "1000")
        )
        picker(element.append(datepicker_input))
        picker.flush()
        return picker
    
    def get_date(self):
        "Extract the date string from the date picker widget."
        picker = self.date_picker
        element = picker.element()
        return picker.evaluate(element.children(":first").val())
    
    def make_modal_dialog(self):
        "Make a modal dialog as a js_proxy widget."
        dialog = js_proxy.ProxyWidget()
        element = dialog.element()
        dialog(element.dialog({"modal": True}).html("No message yet.").dialog("close"))
        dialog.flush()
        return dialog
    
    def warning(self, message):
        "Warn the user of an error using the modal dialog."
        dialog = self.dialog
        element = dialog.element()
        dialog(element.html(message).dialog("open"))
        dialog.flush()

# Create the application object.
app = ToDoApp()

# Display the application interface.
app.show()

# Explore the application state
from jp_gene_viz.examine import examine
examine(app)



