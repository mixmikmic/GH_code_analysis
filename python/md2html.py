import bleach
from markdown import markdown

def htmlize(text):
    """
    This helper method renders Markdown then uses Bleach to sanitize it as
    well as converting all links in text to actual anchor tags.
    """
    text = bleach.clean(text, strip=True)    # Clean the text by stripping bad HTML tags
    text = markdown(text)                    # Convert the markdown to HTML
    text = bleach.linkify(text)              # Add links from the text and add nofollow to existing links

    return text

md = """
# My Markdown Document

For more information, search on [Google](http://www.google.com).

_Grocery List:_

1. Apples
2. Bananas
3. Oranges

"""

print(htmlize(md))



