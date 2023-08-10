from collections import UserList
from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
import CommonMark
from IPython.core.inputsplitter import IPythonInputSplitter
from IPython.core.inputtransformer import InputTransformer
from textwrap import indent, dedent

def codify(source, parser=CommonMark.Parser, str="""""") -> str:
    """Convert markdown to code with the correct line numbers and positions in a string.

    This function replaces non-code_block nodes with blanklines."""
    for node, _ in parser().parse(source).walker():
        if node.t == 'code_block':
            line, col = node.sourcepos[0]
            end = line + len(node.literal.splitlines())
            lines = node.literal.splitlines()
            while len(str.splitlines()) < (line-1): str += '\n'
            str += indent(node.literal, ' '*col)
    return str

def display(source)->'display':
    """Support front matter syntax in cells."""
    metadata, markdown = """""", source
    if source.lstrip().startswith('---'):
        if '\n---' in source.lstrip():
            _, metadata, markdown = source.split('---', 2)
        
    from IPython.display import Markdown, display
    from yaml import safe_load
    metadata = safe_load(metadata)
    if isinstance(metadata, str) ^ (not metadata): markdown, metadata = source, {}
    Markdown.__repr__ = Markdown._repr_markdown_
    display(Markdown(markdown), metadata=metadata)

class Markdown(UserList, InputTransformer):
    def push(self, object): 
        return self.append(object)
   
    def load(self):
        ip = get_ipython() or InteractiveShell()
        if ip.input_transformer_manager.physical_line_transforms:
            self.input_transformer_manager = ip.input_transformer_manager
            ip.input_transformer_manager = IPythonInputSplitter(
                logical_line_transforms=[], physical_line_transforms=[], python_line_transforms=[])
        else:
            self.input_transformer_manager = IPythonInputSplitter()
        ip.input_transformer_manager.python_line_transforms = [
            object for object in ip.input_transformer_manager.python_line_transforms
            if not isinstance(object, Markdown)
        ] + [self]
        return self

    def reset(self):
        str, self.data = '\n'.join(self), []
        if str and str.splitlines()[0].strip():
            display(str)
        return dedent(self.input_transformer_manager.transform_cell(codify(str)))

from nbconvert.preprocessors import Preprocessor
from traitlets import Int
class Normalize(Preprocessor):
    characters = Int(default_value=100)
    def preprocess_cell(Normalize, cell, resources={}, index=0):
        import black
        if cell['cell_type'] == 'code':
            cell.source = black.format_str(Markdown(cell['source']).reset(), Normalize.characters)
        return cell, resources

def unload_ipython_extension(ip=get_ipython()):
    for object in ip.input_transformer_manager.python_line_transforms:
        if isinstance(object, Markdown):
            ip.input_transformer_manager = object.input_transformer_manager
            break


def load_ipython_extension(ip=None): Markdown().load()

class Test(__import__('unittest').TestCase): 
    def setUp(Test):
        get_ipython().run_line_magic('reload_ext', 'pidgin')
        load_ipython_extension()
        from nbformat import write, v4
        with open('test_markdown.ipynb', 'w') as file:
            write(v4.new_notebook(cells=[v4.new_code_cell("""Some paragraph\n\n    a=42""")]), file)
            
    def runTest(Test):
        global test_markdown
        import test_markdown
        assert test_markdown.__file__.endswith('.ipynb')
        assert test_markdown.a is 42
        
    def tearDown(Test):
        get_ipython().run_line_magic('unload_ext', 'pidgin')
        get_ipython().run_line_magic('rm', 'test_markdown.ipynb')
        unload_ipython_extension()

if __name__ == '__main__': 
    __import__('unittest').TextTestRunner().run(Test())



