data = '''
<xml_data>
    <person>
        <id>01</id>
        <name>
            <first>Hrant</first>
            <last>Davtyan</last>
        </name>
        <status organization="AUA">Instructor</status>
    </person>
    <person>
        <id>02</id>
        <name>
            <first>Jack</first>
            <last>Nicolson</last>
        </name>
        <status organization="Hollywood">Actor</status>
    </person>
</xml_data>
'''

from lxml import etree

tree = etree.fromstring(data)

tree.find('person').text

tree.findall("person/name/last")[1].text

tree.find("person/status").text

tree.find("person/status").get("organization")

for i in tree:
    print("Full name: "+ i.find("name/first").text + i.find("name/last").text)
    print("Position: " + i.find("status").text + " at " + i.find("status").get("organization")+"\n")

with open('output.xml', 'w') as f:
    f.write(etree.tostring(tree, pretty_print = True))

with open('output.xml') as f:
    tree = etree.parse(f)

tree.find('person/id').text

