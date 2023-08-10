get_ipython().magic('load preamble_directives.py')

from source_code_analysis.models import SoftwareProject

projects = SoftwareProject.objects.all()

# Write Coherence Report
def write_coherence_report(coherence_report_filepath, target_methods):
    with open(coherence_report_filepath, 'w') as coherence_report:
        for method in target_methods:
            evaluation = method.agreement_evaluations.all()[0]
            coherence_value = 'COHERENT' if evaluation.agreement_vote in [3, 4] else 'NOT_COHERENT'
            coherence_report.write('{0}, {1}\n'.format(method.pk, coherence_value))

# Write Raw Data Report
def write_raw_data_report(raw_report_filepath, target_methods):
    with open(raw_report_filepath, 'w') as raw_report:
        for method in target_methods:
            software_system_name = method.project.name + method.project.version.replace('.', '')
            raw_report.write('{mid}, {method_name}, {class_name}, {software_system}\n'.format(
                                    mid=method.id, method_name=method.method_name, class_name=method.code_class.class_name,
                                    software_system=software_system_name))
            
            method_fp = method.file_path
            relative_filepath = method_fp[method_fp.find('extracted')+len('extracted')+1:]
            raw_report.write('{filepath}, {start_line}, {end_line}\n'.format(filepath=relative_filepath, 
                                                                             start_line=method.start_line, 
                                                                             end_line=method.end_line))
            
            raw_report.write('{comment_len}\n'.format(comment_len=len(method.comment.splitlines())))
            raw_report.write('{comment}'.format(comment=method.comment))
            if not method.comment.endswith('\n'):
                raw_report.write('\n')
                
            raw_report.write('{code_len}\n'.format(code_len=len(method.code_fragment.splitlines())))
            raw_report.write('{code}'.format(code=method.code_fragment))
            if not method.code_fragment.endswith('\n'):
                raw_report.write('\n')
            
            # Last Line of this method
            raw_report.write('###\n')

RAW_DATA_SUFFIX = 'Raw_Data.txt'
COHERENCE_DATA_SUFFIX = 'Coherence_Data.txt'

import os

# Create Report Folder
report_folderpath = os.path.join(os.path.abspath(os.path.curdir), 'report_files')
if not os.path.exists(report_folderpath):
    os.makedirs(report_folderpath)

all_methods_list = list()

# Project-Specific Reports

for project in projects:
    software_system_name = project.name + project.version.replace('.', '')
    target_methods = list()
    project_methods = project.code_methods.order_by('pk')
    # Collect Project Methods whose evaluations are Coherent|Not Coherent
    for method in project_methods:
        evaluation = method.agreement_evaluations.all()[0]
        if not evaluation.wrong_association and evaluation.agreement_vote != 2:
            target_methods.append(method)
            
    all_methods_list.extend(target_methods)
    
    # Coherence Data Report
    coherence_report_filename = '{0}_{1}'.format(software_system_name, COHERENCE_DATA_SUFFIX)
    coherence_report_filepath = os.path.join(report_folderpath, coherence_report_filename)
    
    write_coherence_report(coherence_report_filepath, target_methods)
    
    # Raw Data Report
    raw_report_filename = '{0}_{1}'.format(software_system_name, RAW_DATA_SUFFIX)
    raw_report_filepath = os.path.join(report_folderpath, raw_report_filename)
    
    write_raw_data_report(raw_report_filepath, target_methods)
    

# -- Entire Benchmark Reports

# Coherence Data Report
coherence_report_filename = '{0}_{1}'.format('Benchmark', COHERENCE_DATA_SUFFIX)
coherence_report_filepath = os.path.join(report_folderpath, coherence_report_filename)

write_coherence_report(coherence_report_filepath, all_methods_list)

# Raw Data Report
raw_report_filename = '{0}_{1}'.format('Benchmark', RAW_DATA_SUFFIX)
raw_report_filepath = os.path.join(report_folderpath, raw_report_filename)

write_raw_data_report(raw_report_filepath, all_methods_list)



