from scripts.open_spec import *

file_path = get_abs_path("data/spectra_example.dat")
print file_path

file_path2 = get_abs_path("data/diff_scan_example.dat")
print file_path2

get_diff_scan(file_path)

get_diff_scan(file_path2)

get_scan_details(file_path)

get_scan_details(file_path2)



