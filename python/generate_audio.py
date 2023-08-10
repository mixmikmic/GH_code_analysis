
modelfn="out/tale-prog-stateful8l/model-505.h5" 
outname="longbabble505b"
c2file="test/TaleOfTwoCities_pt09.c2cb"
seed_start_index = "1169"
generate_length = int(60 / 0.04)

import network_data as nd
import os
from subprocess import check_output, call
from IPython.display import display, Markdown, Audio, Image

os.chdir("/home/ec2-user/store/c2gen")
if os.path.isfile('generated/'+outname+'.wav'):
    print("File", outname, "already exists")
else:  
    print("Starting generator - this could take some time")
    call(["python", "lstm_c2_generation.py", "--generate="+outname, "--generate-len="+str(generate_length), "--seed_index="+seed_start_index, c2file, modelfn])
    print("Generator complete")  
display(Audio(filename="/home/ec2-user/store/c2gen/generated/"+outname+".wav"))  
os.chdir("/home/ec2-user/store/c2gen/notebooks") 
  
nd.plot_gen_audio_waveform(outname) 

