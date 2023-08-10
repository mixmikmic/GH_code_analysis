import vos
datadir="/home/ubuntu/starnet_data/" # or "/path/to/my/starnet/directory"

def starnet_download_file(filename):
    vclient = vos.Client()
    vclient.copy('vos:starnet/public/'+filename, datadir+filename)
    print(filename+' downloaded')



