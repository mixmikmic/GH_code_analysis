from flexx import app, ui, event
app.init_notebook()

b = ui.Button(text='foo')
b

b.text = 'Push me!'

with ui.HBox() as hbox:
    slider = ui.Slider(flex=0)
    progress = ui.ProgressBar(flex=1, value=0.7)
hbox

@slider.connect('value')
def show_slider_value(*events):
    progress.value = slider.value  # or events[-1].new_value

class MyWidget(ui.Widget):
    def init(self):
        with ui.HBox():
            self._slider = ui.Slider(flex=0)
            self._progress = ui.ProgressBar(flex=1)
    
    @event.connect('_slider.value')
    def show_slider_value(self, *events):
        self._progress.value = self._slider.value

w1 = MyWidget()
w1

class MyWidget2(ui.Widget):
    def init(self):
        with ui.HBox():
            self._slider = ui.Slider(flex=0)
            self._progress = ui.ProgressBar(flex=1)
    
    class JS:
        @event.connect('_slider.value')
        def show_slider_value(self, *events):
            self._progress.value = self._slider.value

w2 = MyWidget2()
w2

w3 = app.launch(MyWidget2)

from flexx.ui.examples.drawing import Drawing
Drawing(style='height:100px')  # Draw using the mouse below!

from flexx.ui.examples.twente import Twente
Twente(style='height:300px')

from flexx.ui.examples.drawing import Drawing
from flexx.ui.examples.twente import Twente
from flexx.ui.examples.split import Split
with ui.DockPanel(style='height:300px') as w4:  # min-height does not seem to work very well for panel-based layouts
    Twente(title='Twente', flex=1)
    Drawing(title='Drawing', flex=1)
    Split(title='Split', flex=1)
w4



