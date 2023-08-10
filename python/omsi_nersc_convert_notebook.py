import sys
import os 
import ipywidgets as widgets
from IPython.display import display
import getpass

# Import main BASTet convert tool 
try:
    from omsi.tools.convertToOMSI import main as convert_omsi
except ImportError:
    sys.path.append("/project/projectdirs/openmsi/omsi_processing_status/bastet")
    from omsi.tools.convertToOMSI import main as convert_omsi
    print "We recommend using the OpenMSI Jupyter Python Kernel"
    print "To install the OpenMSI kernel copy the following text to $HOME/.ipython/kernels/openmsi/kernel.json"
    print ""
    print """{
 "argv": [
  "/global/project/projectdirs/openmsi/jupyterhub_libs/anaconda/bin/python",
  "-m",
  "IPython.kernel",
  "-f",
  "{connection_file}"
 ],
 "env": {
    "PATH": "/global/project/projectdirs/openmsi/jupyterhub_libs/anaconda/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
 },
 "display_name": "OpenMSI Python 2 Test",
 "language": "python"
}"""
    
# Jupyter sets up logging so that log message are not displayed in the notebook, so we need to 
# reload the logging module in order to be able to have log messages appear in the notebook
import logging
reload(logging)
from omsi.shared.log import log_helper
log_helper.setup_logging()
log_helper.set_log_level('DEBUG')

#####################################
# Basic settings    
#####################################
username = getpass.getuser()
omsi_original_data = os.path.join("/project/projectdirs/openmsi/original_data", username)
omsi_private_data = os.path.join("/project/projectdirs/openmsi/omsi_data_private", username)

#####################################
# Create the main UI   
#####################################
ui_elements_main = []
ui_checkoptions = []
# Select a file
fileselect_widget = widgets.Select(
    options=os.listdir(omsi_original_data),
    description='Input file:',
    disabled=False
)
ui_elements_main.append(fileselect_widget)
# Add to OpenMSI
addomsi_widget = widgets.Checkbox(
    value=True,
    description='Add to OpenMSI',
    disabled=False
)
ui_checkoptions .append(addomsi_widget)
# Global peak detection
fpg_widget = widgets.Checkbox(
    value=False,
    description='Find Peaks (Global)',
    disabled=False
)
ui_checkoptions .append(fpg_widget)
# Local peak detection
fpl_widget = widgets.Checkbox(
    value=False,
    description='Find Peaks (Local)',
    disabled=False
)
ui_checkoptions .append(fpl_widget)
# Local peak detection
nmf_widget = widgets.Checkbox(
    value=False,
    description='NMF',
    disabled=False
)
ui_checkoptions .append(nmf_widget)
# Local peak detection
tic_widget = widgets.Checkbox(
    value=False,
    description='TIC Normalize',
    disabled=False
)
ui_checkoptions .append(tic_widget)
# EMail notification
email_widget = widgets.Checkbox(
    value=False,
    description='EMail Notification',
    disabled=False
)
email_text_widget = widgets.Text(
    value=username+"@lbl.gov",
    description='Email:',
    disabled=False
)
ui_checkoptions.append(widgets.HBox([email_widget, email_text_widget]))

#ui_checkoptions .append(email_widget)
ui_elements_main.append(widgets.VBox(ui_checkoptions))
main_ui = widgets.HBox(ui_elements_main)

def create_convert_settings():
    settings = ['convertToOMSI.py', 
            '--no-xdmf',
            '--user', username,
            '--regions', 'merge',
            '--db-server', 'https://openmsi.nersc.gov/openmsi/',
            '--compression',
            '--thumbnail',
            '--auto-chunking',
            '--error-handling', 'terminate-and-cleanup',
           ]
    if addomsi_widget.value:
        settings.append('--add-to-db')
    else:
        settings.append('--no-add-to-db')
    if fpl_widget.value:
        settings.append('--fpl')
    else:
        settings.append('--no-fpl')
    if fpg_widget.value:
        settings.append('--fpg')
    else:
        settings.append('--no-fpg')
    if tic_widget.value:
        settings.append('--ticnorm')
    else:
        settings.append('--no-ticnorm')
    if nmf_widget.value:
        settings.append('--nmf')
    else:
        settings.append('--no-nmf')
    if email_widget.value:
        settings.append('--email')
        settings.append('oruebel@lbl.gov'),
        settings.append('bpbowen@lbl.gov'),
        settings.append('email_text_widget.value')
    settings.append(os.path.join(omsi_original_data, fileselect_widget.value))
    settings.append(os.path.join(omsi_private_data, fileselect_widget.value) + '.h5')
    return settings

display(main_ui)

settings=create_convert_settings()
convert_omsi(argv=settings)



