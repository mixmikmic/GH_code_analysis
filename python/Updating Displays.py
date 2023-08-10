from IPython.display import display, update_display

handle = display('x', display_id='update-me')
handle

handle.display('y')

handle.update('z')

handle = display("hello", display_id=True)
handle

display('x', display_id='here');

display('y', display_id='here');

update_display('z', display_id='here')

import os
from binascii import hexlify

class ProgressBar(object):
    def __init__(self, capacity):
        self.progress = 0
        self.capacity = capacity
        self.html_width = '60ex'
        self.text_width = 60
        self._display_id = hexlify(os.urandom(8)).decode('ascii')
        
    def __repr__(self):
        fraction = self.progress / self.capacity
        filled = '=' * int(fraction * self.text_width)
        rest = ' ' * (self.text_width - len(filled))
        return '[{}{}] {}/{}'.format(
            filled, rest,
            self.progress, self.capacity,
        )
    
    def _repr_html_(self):
        return """<progress
            value={progress}
            max={capacity}
            style="width: {width}"/>
            {progress} / {capacity}
        """.format(
            progress=self.progress,
            capacity=self.capacity,
            width=self.html_width,
        )
    
    def display(self):
        display(self, display_id=self._display_id)
    
    def update(self):
        update_display(self, display_id=self._display_id)

bar = ProgressBar(10)
bar.display()

import time

bar.display()

for i in range(11):
    bar.progress = i
    bar.update()
    time.sleep(0.25)

