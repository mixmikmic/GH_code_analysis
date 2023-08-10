import ipywidgets as widgets

def calculate_password(length):
    import string
    import secrets
    
    # Gaenerate a list of random letters of the correct length.
    password = ''.join(secrets.choice(string.ascii_letters) for _ in range(length))

    return password

calculate_password(10)

helpful_title = widgets.HTML('Generated password is:')
password_text = widgets.HTML('No password yet')
password_text.layout.margin = '0 0 0 20px'
password_length = widgets.IntSlider(description='Length of password',
                                   min=8, max=20,
                                   style={'description_width': 'initial'})

password_widget = widgets.VBox(children=[helpful_title, password_text, password_length])
password_widget



def update_password(change):
    length = int(change.new)
    new_password = calculate_password(length)
    
    # NOTE THE LINE BELOW: it relies on the password widget already being defined.
    password_text.value = new_password
    
password_length.observe(update_password, names='value')

password_widget

from ipywidgets import interact
from IPython.display import display
interact(calculate_password, length=(8, 20));

@interact(length=(8, 20))
def print_password(length):
    print(calculate_password(length))



