import ipywidgets as widgets

helpful_title = 0  # Replace with some that displays "Generated password is:"
password_text = 0  # Replace with something that displays "No password set"
password_length = 0 # Replace with slider

password_widget = widgets.VBox(children=[helpful_title, password_text, password_length])
password_widget

# %load solutions/bad-password-generator/bad-pass-pass1-widgets.py

def calculate_password(change):
    import string
    from secrets import choice
    length = change.new
    # Generate a list of random letters of the correct length.
    password = ''.join(choice(string.ascii_letters) for _ in range(length))
    # Add a line below to set the value of the widget password_text

# %load solutions/bad-password-generator/bad-pass-pass1-passgen.py

# call calculate_password whenever the password length changes

# %load solutions/bad-password-generator/bad-pass-pass1-observe.py

