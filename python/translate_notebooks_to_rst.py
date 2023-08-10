import glob
import os.path

notebooks = glob.glob('source/analyses/*/*.ipynb')
notebooks += glob.glob('source/data/*/*.ipynb')
notebooks

for infile in notebooks:
    infile = infile.replace(' ','\ ')
    
    #convert notebook in its local directory
    path, nbfile = os.path.split(infile)    
    get_ipython().system(' cd $path; ipython nbconvert --to rst $nbfile')

    #move the .rst file
    no_ext = os.path.splitext(nbfile)[0]
    movefiles = path+'/'+no_ext+'.rst'
    dest = 'docs/'+'/'.join(path.split('/')[1:]) #pretty dependent on directory structure
    get_ipython().system(' mv $movefiles $dest')

    #move supporting files, if there are any
    movedir = path+'/'+no_ext+'_files'
    if os.path.exists(movedir):
        destdir = 'docs/'+'/'.join(path.split('/')[1:])+'/'+no_ext+'_files'        
        if os.path.exists(destdir): #check to see if the 'files' directory has already been created
            get_ipython().system('rm -r $destdir #if so, remove it, so we can have a clean slate...')
        get_ipython().system('mv $movedir $dest')
        
    print '\n'



