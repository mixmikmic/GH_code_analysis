with open('../../cgetools/assets/map_template.jinja', 'r') as f:
    template = Template(f.read())

# Update these to change the text
template_variables = {
    'title': TITLE,
    'narrative': 'Some explanatory text.',
    'tooltip_css': '<link href="../../cgetools/assets/tooltip.css" rel="stylesheet" type="text/css">',
}

# Use inline resources, render the html and open
html = file_html(layout, resources=INLINE, title=TITLE, template=template, template_variables=template_variables)


# Uncomment the next two lines if you'd like to save the file
#with open('interactive_map_with_slider.html', 'w') as f:
#    f.write(html)

display_html(html, raw=True)

