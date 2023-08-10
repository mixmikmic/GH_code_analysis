import sys, os
sys.path.append('./modules')
from Monitoring_Tools import *
from OpenStack_Tools  import *

platform=os.getenv('OS_PLATFORM', 'demo8')
display_platform(platform)
inventory = read_inventory("$HOME/env/{}_hosts.ini".format(platform))
conn = connectToCloud(platform)
#inventory

HTML_URL = show_notebook_url(platform, host_ip="10.3.216.210", port=8888)

display_html_ping_all(inventory)

displayServerList( conn )

display_html_ping_ports_all(inventory)

display_html_endpoint_urls(conn)

for host in sorted(inventory['ssh_check']):    
    ip = inventory['hosts'][host]['ansible_host']
    user = inventory['hosts'][host]['ansible_user']
    pkey = inventory['hosts'][host]['ssh_key']
    
    stdout, stderr = ssh_command(host, ip, user, pkey, "uptime")
    
    #print("LINE=" + stdout)
    uptime = strip_uptime(stdout)
    print(host + ":" + uptime)

get_ipython().system(' [ ! -d history ] && mkdir history')

import datetime
import time

#d = datetime.date.today().strftime("%B %d, %Y")
#dt = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
d = datetime.date.today().strftime("%Y-%m-%d")
dt = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
print(d)
print(dt)

for host in sorted(inventory['df_check']):    
    ip = inventory['hosts'][host]['ansible_host']
    user = inventory['hosts'][host]['ansible_user']
    pkey = inventory['hosts'][host]['ssh_key']
    df_check = inventory['hosts'][host]['df_check']
    
    # write to history subdir (~/notebooks/cron for cron jobs)
    history_file='history/df_history_' + platform + '_' + host + '.txt'
    history_fd = open(history_file, 'a')
    
    full_df_cmd="hostname; df 2>&1"
    df_op, stderr = ssh_command(host, ip, user, pkey, full_df_cmd)    
    history_fd.write('DATE:' + dt + '\n' + df_op)
    history_fd.close()
    
    df_cmd="df " + df_check.replace(",", " ") + "| grep -v ^Filesystem"
    df_op, stderr = ssh_command(host, ip, user, pkey, df_cmd)    
    #df_op = stdout.decode('utf8')
    #print("HOST[" + host + "]<" + df_check + ">{" + df_cmd +"}:" + df_op)
    
    df_lines=df_op.split("\n")
    for df_line in df_lines:
        #print("LINE: " + df_line)
        pc_pos = df_line.find("%")
        if pc_pos != -1:
            pc=int(df_line[pc_pos-3:pc_pos])
            partn=df_line[pc_pos+1:]
            print(host + " " + str(pc) + "% " + partn)

get_ipython().magic('matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 3*np.pi, 500)
plt.plot(x, np.sin(x**2))
plt.title('A simple chirp');

from ipywidgets import interactive
from IPython.display import Audio, display
import numpy as np

get_ipython().run_cell_magic('html', '', '\n<style>\n     .pieContainer {\n          height: 260px;\n          position: relative;\n     }\n     .pieBackground {\n          background-color: lightgrey;\n          position: relative;\n          width: 180px;\n          height: 180px;\n          -moz-border-radius: 90px;\n          -webkit-border-radius: 90px;\n          -o-border-radius: 90px;\n          border-radius: 90px;\n          -moz-box-shadow: -1px 1px 3px #000;\n          -webkit-box-shadow: -1px 1px 3px #000;\n          -o-box-shadow: -1px 1px 3px #000;\n          box-shadow: -1px 1px 3px #000;\n     } \n     .pie {\n          position: absolute;\n          width: 180px;\n          height: 180px;\n          -moz-border-radius: 90px;\n          -webkit-border-radius: 90px;\n          -o-border-radius: 90px;\n          border-radius: 90px;\n          clip: rect(0px, 90px, 180px, 0px);\n     }\n     .hold {\n          position: absolute;\n          width: 180px;\n          height: 180px;\n          -moz-border-radius: 90px;\n          -webkit-border-radius: 90px;\n          -o-border-radius: 90px;\n          border-radius: 90px;\n          clip: rect(0px, 180px, 180px, 90px);\n     }\n     #pieSliceX10 .pie {\n          background-color: #1b458b;\n          -webkit-transform:rotate(10deg);\n          -moz-transform:rotate(10deg);\n          -o-transform:rotate(10deg);\n          transform:rotate(10deg);\n     }\n</style>\n\n<div class="pieContainer">\n     <h3>Disk usage</h3><p/>\n     <div class="pieBackground"> <div id="pieSliceX10" class="hold"><div class="pie"></div></div></div>\n    \n</div>')

get_ipython().run_cell_magic('html', '', '\n<!-- ADAPTED from https://codepen.io/AtomicNoggin/pen/fEish -->\n    \n<div class="pieContainer2">\n  <div class="pie2" data-start="0" data-value="30"></div>\n  <div class="pie2 highlight" data-start="30" data-value="30"></div>\n  <div class="pie2" data-start="60" data-value="40"></div>\n  <div class="pie2 big" data-start="100" data-value="260"></div>\n</div>\n\n<style>\n     .pieContainer2 {\n          height: 260px;\n          position: relative;\n     }\n\n.pie2 {\n\t\tposition:absolute;\n\t\twidth:100px;\n\t\theight:200px;\n\t\toverflow:hidden;\n\t\tleft:150px;\n\t\t-moz-transform-origin:left center;\n\t\t-ms-transform-origin:left center;\n\t\t-o-transform-origin:left center;\n\t\t-webkit-transform-origin:left center;\n\t\ttransform-origin:left center;\n\t}\n/*\n  unless the piece represents more than 50% of the whole chart.\n  then make it a square, and ensure the transform origin is\n  back in the center.\n\n  NOTE: since this is only ever a single piece, you could\n  move this to a piece specific rule and remove the extra class\n*/\n\t.pie2.big {\n\t\twidth:200px;\n\t\theight:200px;\n\t\tleft:50px;\n\t\t-moz-transform-origin:center center;\n\t\t-ms-transform-origin:center center;\n\t\t-o-transform-origin:center center;\n\t\t-webkit-transform-origin:center center;\n\t\ttransform-origin:center center;\n\t}\n/*\n  this is the actual visible part of the pie. \n  Give it the same dimensions as the regular piece.\n  Use border radius make it a half circle.\n  move transform origin to the middle of the right side.\n  Push it out to the left of the containing box.\n*/\n\t.pie2:BEFORE {\n\t\tcontent:"";\n\t\tposition:absolute;\n\t\twidth:100px;\n\t\theight:200px;\n\t\tleft:-100px;\n\t\tborder-radius:100px 0 0 100px;\n\t\t-moz-transform-origin:right center;\n\t\t-ms-transform-origin:right center;\n\t\t-o-transform-origin:right center;\n\t\t-webkit-transform-origin:right center;\n\t\ttransform-origin:right center;\n\t\t\n\t}\n /* if it\'s part of a big piece, bring it back into the square */\n\t.pie2.big:BEFORE {\n\t\tleft:0px;\n\t}\n/* \n  big pieces will also need a second semicircle, pointed in the\n  opposite direction to hide the first part behind.\n*/\n\t.pie2.big:AFTER {\n\t\tcontent:"";\n\t\tposition:absolute;\n\t\twidth:100px;\n\t\theight:200px;\n\t\tleft:100px;\n\t\tborder-radius:0 100px 100px 0;\n\t}\n/*\n  add colour to each piece.\n*/\n\t.pie2:nth-of-type(1):BEFORE,\n\t.pie2:nth-of-type(1):AFTER {\n\t\tbackground-color:blue;\t\n\t}\n\t.pie2:nth-of-type(2):AFTER,\n\t.pie2:nth-of-type(2):BEFORE {\n\t\tbackground-color:green;\t\n\t}\n\t.pie2:nth-of-type(3):AFTER,\n\t.pie2:nth-of-type(3):BEFORE {\n\t\tbackground-color:red;\t\n\t}\n\t.pie2:nth-of-type(4):AFTER,\n\t.pie2:nth-of-type(4):BEFORE {\n\t\tbackground-color:orange;\t\n\t}\n/*\n  now rotate each piece based on their cumulative starting\n  position\n*/\n\t.pie2[data-start="30"] {\n\t\t-moz-transform: rotate(30deg); /* Firefox */\n\t\t-ms-transform: rotate(30deg); /* IE */\n\t\t-webkit-transform: rotate(30deg); /* Safari and Chrome */\n\t\t-o-transform: rotate(30deg); /* Opera */\n\t\ttransform:rotate(30deg);\n\t}\n\t.pie2[data-start="60"] {\n\t\t-moz-transform: rotate(60deg); /* Firefox */\n\t\t-ms-transform: rotate(60deg); /* IE */\n\t\t-webkit-transform: rotate(60deg); /* Safari and Chrome */\n\t\t-o-transform: rotate(60deg); /* Opera */\n\t\ttransform:rotate(60deg);\n\t}\n\t.pie2[data-start="100"] {\n\t\t-moz-transform: rotate(100deg); /* Firefox */\n\t\t-ms-transform: rotate(100deg); /* IE */\n\t\t-webkit-transform: rotate(100deg); /* Safari and Chrome */\n\t\t-o-transform: rotate(100deg); /* Opera */\n\t\ttransform:rotate(100deg);\n\t}\n/*\n  and rotate the amount of the pie that\'s showing.\n\n  NOTE: add an extra degree to all but the final piece, \n  to fill in unsightly gaps.\n*/\n\t.pie2[data-value="30"]:BEFORE {\n\t\t-moz-transform: rotate(31deg); /* Firefox */\n\t\t-ms-transform: rotate(31deg); /* IE */\n\t\t-webkit-transform: rotate(31deg); /* Safari and Chrome */\n\t\t-o-transform: rotate(31deg); /* Opera */\n\t\ttransform:rotate(31deg);\n\t}\n\t.pie2[data-value="40"]:BEFORE {\n\t\t-moz-transform: rotate(41deg); /* Firefox */\n\t\t-ms-transform: rotate(41deg); /* IE */\n\t\t-webkit-transform: rotate(41deg); /* Safari and Chrome */\n\t\t-o-transform: rotate(41deg); /* Opera */\n\t\ttransform:rotate(41deg);\n\t}\n\t.pie2[data-value="260"]:BEFORE {\n\t\t-moz-transform: rotate(260deg); /* Firefox */\n\t\t-ms-transform: rotate(260deg); /* IE */\n\t\t-webkit-transform: rotate(260deg); /* Safari and Chrome */\n\t\t-o-transform: rotate(260deg); /* Opera */\n\t\ttransform:rotate(260deg);\n\t}\n\n</style> ')

# TODO: 
# equivalent of openstack service list

# See: how service list is implemented:
#    https://github.com/openstack/python-openstackclient/blob/master/openstackclient/compute/v2/service.py
# (compute_client = client_manager.compute)
#
# See also: how to create/use a clientmanager:
#    https://github.com/openstack/python-openstackclient/blob/master/examples/osc-lib.py

#from openstackclient.common import clientmanager
#from openstackclient.common import utils

from IPython.display import display
from IPython.display import HTML
import IPython.core.display as di

# This line will hide code by default when the notebook is exported as HTML
di.display_html('<script>jQuery(function() {if (jQuery("body.notebook_app").length == 0) { jQuery(".input_area").toggle(); jQuery(".prompt").toggle();}});</script>', raw=True)

# This line will add a button to toggle visibility of code blocks, for use with the HTML export version
di.display_html('''<button onclick="jQuery('.input_area').toggle(); jQuery('.prompt').toggle();">Toggle code</button>''', raw=True)

import IPython

IPython.display.HTML('<h1>Platform status(es)</h1><div id="status"></div>')



from IPython.display import Javascript

js = """
    $('#status').html('');

"""
js = Javascript(js)
display(js)

for platform in ['demo8', 'poc1', 'poc2', 'nfv5']:
    #platform=os.getenv('OS_PLATFORM', 'demo8')
    #print(platform)
    js = """
    console.log('plaform=""" + platform + """');
    $('#status').html($('#status').html() + '<h2>Platform:""" + platform + """</h2>');

"""
    js = Javascript(js)
    display(js)



