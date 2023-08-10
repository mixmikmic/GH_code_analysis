symbTrs = []

newmakam = None
newform = None
newusul = None
newname = None
newcomposer = None

from fileoperations import get_filenames_in_dir
import os
import json

symbTrfolder = os.path.abspath(os.path.join('..','..'))
symbTrTxtfolder = os.path.join(symbTrfolder, 'txt/')
symbTrPdffolder = os.path.join(symbTrfolder, 'SymbTr-pdf/')
symbTrMu2folder = os.path.join(symbTrfolder, 'mu2/')
symbTrXmlfolder = os.path.join(symbTrfolder, 'MusicXML/')
symbTrMidfolder = os.path.join(symbTrfolder, 'midi/')

folders = [symbTrTxtfolder, symbTrPdffolder, symbTrMu2folder, symbTrXmlfolder, symbTrMidfolder]
extensions = ['txt', 'pdf', 'mu2', 'xml', 'mid']

symbTr_work_file = os.path.join(symbTrfolder, 'symbTr_mbid.json')

scriptfolder = os.path.join(symbTrfolder, 'extras')
sep = '--'


for symbTr in symbTrs:
    sybmTrattributes = symbTr.split(sep)
    
    if newmakam is not None:
        sybmTrattributes[0] = newmakam 
    if newform is not None:
        sybmTrattributes[1] = newform
    if newusul is not None:
        sybmTrattributes[2] = newusul
    if newname is not None:
        sybmTrattributes[3] = newname
    if newcomposer is not None:
        sybmTrattributes[4] = newcomposer

    newsymbTr = sep.join(sybmTrattributes)
    
    commitstr = 'git commit -m "Changed the name of ' + symbTr + ' -> ' + newsymbTr + '"'
    for f, e in zip(folders, extensions):
        os.chdir(f)
        mvstr = 'git mv ' + symbTr + '.' + e + ' ' + newsymbTr + '.' + e
        os.system(mvstr)

        if e == 'pdf':  # commit changes to SymbTrpdf submodule
            os.system(commitstr)

    worksymbtr = json.load(open(symbTr_work_file, 'r'))
    for ws in worksymbtr:
        if ws['name'] == symbTr:
            ws['name'] = newsymbTr
            print ws
    worksymbtr = json.dump(worksymbtr, open(symbTr_work_file, 'w'))
    
    os.chdir(symbTrfolder)

    addstr = 'git add symbTr_mbid.json SymbTr-pdf'
    os.system(addstr)
    os.system(commitstr)
    os.chdir(scriptfolder)

