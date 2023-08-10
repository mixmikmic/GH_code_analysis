# First import the compliance checker and test that it is installed properly.
from compliance_checker.runner import ComplianceChecker, CheckSuite

# Load all available checker classes.
check_suite = CheckSuite()
check_suite.load_all_available_checkers()

# Path to the Scripps Pier Data.

buoy_path = 'https://data.nodc.noaa.gov/thredds/dodsC/ioos/sccoos/scripps_pier/scripps_pier-2016.nc'

output_file = 'buoy_testCC.txt'

return_value, errors = ComplianceChecker.run_checker(
    ds_loc=buoy_path,
    checker_names=['cf', 'acdd'],
    verbose=True,
    criteria='normal',
    skip_checks=None,
    output_filename=output_file,
    output_format='text'
)

with open(output_file, 'r') as f:
    print(f.read())

