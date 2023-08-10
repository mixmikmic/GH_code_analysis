# Build the XML tree from scratch
from xml.etree import ElementTree
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement

# <membership/>
membership = Element( 'membership' )

# <membership><users/>
users = SubElement( membership, 'users' )

# <membership><users><user/>
SubElement( users, 'user', name='john' )
SubElement( users, 'user', name='charles' )
SubElement( users, 'user', name='peter' )

# <membership><groups/>
groups = SubElement( membership, 'groups' )

# <membership><groups><group/>
group = SubElement( groups, 'group', name='users' )

# <membership><groups><group><user/>
SubElement( group, 'user', name='john' )
SubElement( group, 'user', name='charles' )

# <membership><groups><group/>
group = SubElement( groups, 'group', name='administrators' )

# <membership><groups><group><user/>
SubElement( group, 'user', name='peter' )

output_file = open( 'data/membership.xml', 'w' )
output_file.write( '<?xml version="1.0"?>' )
output_file.write( ElementTree.tostring( membership ) )
output_file.close()

from xml.etree import ElementTree

document = ElementTree.parse( 'data/membership.xml' )
membership = document.getroot()
users = membership.find( 'users' )

for user in document.findall( 'users/user' ):
    print user.attrib[ 'name' ]

for group in document.findall( 'groups/group' ):
    print group.attrib[ 'name' ]

for group in document.findall( 'groups/group' ):
    print 'Group:', group.attrib[ 'name' ]
    print 'Users:'
    for node in list(group):
        if node.tag == 'user':
            print '-', node.attrib[ 'name' ]

users = document.find( 'users' )
for node in users.getiterator():
    print node.tag, node.attrib, node.text, node.tail

users = document.find( 'users' )
for node in list(users):
    print node.tag, node.attrib, node.text, node.tail





















