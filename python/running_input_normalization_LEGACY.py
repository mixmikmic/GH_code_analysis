import qtools
import pandas as pd

# We'll need an input normalized manifest that looks like this:
input_norm_manifest = '/projects/ps-yeolab3/bay001/tbos/input_norm/rbfox2.txt'
output_dir = '/projects/ps-yeolab3/bay001/tbos/input_norm/rbfox2_input_norm/'
df = pd.read_table(input_norm_manifest)
df['INPUT'].ix[0]



command = 'perl /home/elvannostrand/data/clip/CLIPseq_analysis/scripts/LABshare/FULL_PIPELINE_WRAPPER.pl {} {} hg19'.format(
    input_norm_manifest, output_dir)
print command

jobname = 'input_normalization_'
qtools.Submitter(command, jobname, array=False, nodes=1, ppn=1, walltime='2:00:00', submit=True)



