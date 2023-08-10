import logging
logging.getLogger("paramiko").setLevel(logging.WARNING)

from jnpr.junos import Device

from lxml import etree

#dev = Device(host='xxxx', user='demo', password='demo123', gather_facts=False)
dev = Device(host='xxxx', user='demo', password='demo123')

dev.open()

dev.facts



op = dev.rpc.get_interface_information()
#op = dev.rpc.get_interface_information({'format': 'text'})
#op = dev.rpc.get_interface_information(interface_name='lo0', terse=True)

print etree.tostring(op)

print dev.cli("show version", warning=False)

cnf = dev.rpc.get_config()

print etree.tostring(cnf)

data = dev.rpc.get_config(filter_xml=etree.XML('<configuration><system><services/></system></configuration>'))

print etree.tostring(data)

from jnpr.junos.utils.fs import FS

fs = FS(dev)

fs.ls('/var/tmp')

print fs.cat('/var/tmp/nitin.log')

from jnpr.junos.op.routes import RouteTable
tbl = RouteTable(dev)
tbl.get()

for item in tbl:
    print 'protocol:', item.protocol
    print 'age:', item.age
    print 'via:', item.via
    print

from jnpr.junos.factory.factory_loader import FactoryLoader
import yaml

yaml_data="""
---
ArpTable:
  rpc: get-arp-table-information
  item: arp-table-entry
  key: mac-address
  view: ArpView

ArpView:
  fields:
    mac_address: mac-address
    ip_address: ip-address
    interface_name: interface-name
"""
globals().update(FactoryLoader().load(yaml.load(yaml_data)))
arps = ArpTable(dev)
arps.get()
for arp in arps:
        print 'mac_address: ', arp.mac_address
        print 'ip_address: ', arp.ip_address
        print 'interface_name:', arp.interface_name
        print

from jnpr.junos.utils.config import Config
cu = Config(dev)

data = """interfaces { 
    ge-1/0/1 {
        description "MPLS interface";
        unit 0 {
            family mpls;
        }      
    } 
    ge-1/0/2 {
        description "MPLS interface";
        unit 0 {
            family mpls;
        }      
    }   
}
protocols {
    mpls { 
        interface ge-1/0/1; 
        interface ge-1/0/2;            
    }
}
"""

cu.load(data, format='text')
cu.commit_check()

get_ipython().magic('pinfo cu.load')

data = """<policy-options>
          <policy-statement action="delete">
            <name>F5-in</name>
            <term>
                <name>test</name>
                <then>
                    <accept/>
                </then>
            </term>
            <from>
                <protocol>mpls</protocol>
            </from>
        </policy-statement>
        </policy-options>"""


cu.load(data)
cu.commit_check()

xml_temp="""<policy-options>
        <policy-statement>
            <name>all-local</name>
            {% for prot in protocols %}
            <term>
                <name>{{ prot['name'] }}</name>
                <from>
                    <protocol>{{ prot.protocol }}</protocol>
                </from>
                <then>
                    <accept/>
                </then>
            </term>{% endfor %}
        </policy-statement>
    </policy-options>"""

from jinja2 import Template
tmpl = Template(xml_temp)
conf = tmpl.render(protocols=[{'name':'1', 'protocol':'direct'}, {'name':'2', 'protocol':'static'}])
print conf
#cu.load(str(conf))

get_ipython().magic('cat /Users/nitinkr/Coding/pyezex/protocol.conf')

get_ipython().magic('cat /Users/nitinkr/Coding/pyezex/protocol_data.yml')

import yaml
data = yaml.load(open('/Users/nitinkr/Coding/pyezex/protocol_data.yml'))
print data

from jinja2 import Template
tmpl = Template(open('/Users/nitinkr/Coding/pyezex/protocol.conf').read())
conf = tmpl.render(data)
print conf

cu.load(template_path='/Users/nitinkr/Coding/pyezex/protocol.conf',
        template_vars=data, format='text')

cu.pdiff()

cu.rollback()

cu.pdiff()

from jnpr.junos.utils.sw import SW
sw = SW(dev)

get_ipython().magic('pinfo sw.install')

dev = Device(host='xxxx', user='demo', password='demo123', gather_facts=False)
dev.open()
sw = SW(dev)
ok = sw.install(package=r'/Users/nitinkr/Downloads/jinstall-xxxxx-domestic.tgz', progress=update_progress)
if ok:
    print 'rebooting'
    sw.reboot()

from jnpr.junos.utils.scp import SCP

get_ipython().magic('cat info.txt')

get_ipython().magic('rm info.txt')

import time
from bokeh.plotting import figure, output_server, cursession, show
from bokeh.models import NumeralTickFormatter

from jnpr.junos import Device

# prepare output to server
output_server("animated_line")

p = figure(plot_width=600, plot_height=600)
dev = Device(host='xxxx', user='demo', password='demo123', gather_facts=False, port=22)
dev.open()

x_tmp = [0]*5
x_var = [0]*5
ct = time.localtime()
ct = ct.tm_hour*3600+ct.tm_min*60+ct.tm_sec
op = dev.rpc.get_statistics_information(tcp=True)
packets_sent_new = op.xpath('.//packets-sent')[0].text.strip()
packets_recv_new = op.xpath('.//packets-received')[0].text.strip()
p.line([ct, ct+2, ct+4, ct+6, ct+8], x_tmp, name='ex_line',  legend = 'packets-sent')
p.line([ct, ct+2, ct+4, ct+6, ct+8], x_var, name='ex_line', line_color="red", legend = 'packets-recv')
p.xaxis[0].formatter = NumeralTickFormatter(format='00:00:00')
show(p)

# create some simple animation..
# first get our figure example data source
renderer = p.select(dict(name="ex_line"))
ds1 = renderer[0].data_source
ds2 = renderer[1].data_source
while True:
    op = dev.rpc.get_statistics_information(tcp=True)
    packets_sent_new, packets_sent_old = op.xpath('.//packets-sent')[0].text.strip(), packets_sent_new
    packets_recv_new, packets_recv_old = op.xpath('.//packets-received')[0].text.strip(), packets_recv_new
    ct = time.localtime()
    ct = ct.tm_hour*3600+ct.tm_min*60+ct.tm_sec
    ds2.data["x"] = ds1.data["x"] = [ct, ct+2, ct+4, ct+6, ct+8]
    ds1.data["y"] = ds1.data["y"][1:]+[int(packets_sent_new)-int(packets_sent_old)]
    ds2.data["y"] = ds2.data["y"][1:]+[int(packets_recv_new)-int(packets_recv_old)]
    cursession().store_objects(ds1, ds2)
    time.sleep(1.5)

