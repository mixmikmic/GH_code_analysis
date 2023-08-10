class Quantity(object):
    def __init__(self, value, units=None):
        self.value = value
        self.units = units
    
    def __repr__(self):
        return "{} {}".format(self.value, self.units)
    
    def _repr_html_(self):
        html = "${0}\ \mathrm{{{1}}}$".format(self.value, self.units)
        return html

vp= 2500
vp

v = Quantity(2650, 'm/s')
v

class PhoneTable(object):
    """
    Initialize with a dictionary of properties. And does a nice
    html rendered output.
    """

    def __init__(self, properties):
        for k, v in properties.items():
            if k and v:
                setattr(self, k, v)

    def _repr_html_(self):
        """
        IPython Notebook magic repr function.
        """
        rows = ''
        s = '<tr><td><strong>{k}</strong></td><td>{v}</td></tr>'
        for k, v in self.__dict__.items():
            rows += s.format(k=k, v=v)
        html = '<table>{}</table>'.format(rows)
        return html

my_pals = {'Joe':'555-2764', 'Jerry':'555-1123', 'Suzie':'555-1234'}
my_pals

pretty_pals = PhoneTable(my_pals)
pretty_pals



