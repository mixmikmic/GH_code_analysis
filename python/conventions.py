def yaml(str):
    if str.lstrip().startswith('---'):
        return"""locals().update(dict(
            sum((
                list(dict.items()) for dict in __import__('yaml').safe_load_all('''{}''')
                ), [])))""".format(str.lstrip())

def graphviz(str):
    if str.strip().startswith('graph') or str.strip().startswith('digraph'):
        return"""__import__('IPython').display.display(
            __import__('graphviz').Source('''{}''', format="png"))""".format(str)

def iframe(str):
    if str.strip().startswith('http'):
        return"""__import__('IPython').display.display(
        __import__('IPython').display.IFrame(
        \'\'\'{}\'\'\', 800, 600))""".format(str.replace("en.wikipedia.org", "en.m.wikipedia.org").strip())

def doctest(str):
    if str.strip().startswith('>>>'):
        return"""print(__import__('doctest').DocTestRunner().run(__import__('doctest').DocTest(
        [example for example in __import__('doctest').DocTestParser().parse(\'\'\'{0}\'\'\') if not isinstance(example, str)],
        __import__(__name__).__dict__, '<mumble>', None, None, \'\'\'{0}\'\'\'
    )))"""
choices = yaml, graphviz, iframe, doctest

from IPython.core.interactiveshell import InteractiveShell

from IPython import get_ipython
from nbconvert.exporters.templateexporter import TemplateExporter
from IPython.core.inputtransformer import InputTransformer
from jinja2 import Environment
from collections import UserList
from dataclasses import dataclass, field

@dataclass
class Conventions(InputTransformer, UserList):
    data: list = field(default_factory=list)
    def push(self, line): self.data.append(line)
    
    def reset(self, *, str=""""""):
        global choices
        from inspect import getfullargspec
        str, self.data = '\n'.join(self.data), []
        for callable in choices:
            result = callable(str)
            if result is not None:
                str = result
                break
        return str

    def load(self, *args): 
        ip=get_ipython()                 
        self.unload()
        ip.input_transformer_manager.python_line_transforms +=  [self]
        
    def unload(self, *args): 
        ip = (get_ipython() or InteractiveShell())
        ip.input_transformer_manager.python_line_transforms = [
            object for object in ip.input_transformer_manager.python_line_transforms
            if not isinstance(object, Conventions)
        ]

def load_ipython_extension(ip=None): Conventions().load()

def unload_ipython_extension(ip=None): Conventions().unload()

if __name__ == '__main__': load_ipython_extension()



