
get_ipython().system('pip install --upgrade --quiet youtube_dl')


get_ipython().system('pip install --upgrade --quiet PyDrive')


# https://github.com/adaptlearning/adapt_authoring/wiki/Installing-FFmpeg
get_ipython().system('ffmpeg -version')


get_ipython().system('start %windir%\\explorer.exe "C:\\Users\\dev\\Documents\\repositories\\notebooks\\Miscellaneous\\json"')


get_ipython().system('"C:\\Program Files\\Notepad++\\notepad++.exe" "C:\\Users\\dev\\Anaconda3\\Lib\\site-packages\\pydrive\\files.py"')


import youtube_dl

downloads_folder = r'C:\Users\dev\Downloads\\'
youtube_url_list = ['https://d18ky98rnyall9.cloudfront.net/38sc2nxEEeeR4BLAuMMnkA.processed/full/540p/index.mp4?Expires=1523232000']

ydl_opts = {
    'format': 'bestaudio/best',
    'nocheckcertificate': False,
    'outtmpl': downloads_folder+youtube_dl.DEFAULT_OUTTMPL,
    'postprocessors': [{'key': 'FFmpegExtractAudio',
                       'preferredcodec': 'mp3',
                       'preferredquality': '192'}],
    'verbose': False,
    }
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    try:
        result = ydl.download(youtube_url_list)
    except Exception as e:
        print(e)
print('Conversion completed.')


from pydrive.auth import GoogleAuth

# The URL that gets displeyed here is not suitable for public consumption
GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = r'../json/client_secrets.json'
gauth = GoogleAuth()
gauth.LocalWebserverAuth()


from pydrive.drive import GoogleDrive

# Create GoogleDrive instance with authenticated GoogleAuth instance
drive = GoogleDrive(gauth)

# ID of the "Reading Material" folder
tgt_folder_id = '1syfUx6jukbW1CWIEy8xoM9veGGr5MBUh'


import os

# Upload all mp3s from the Downloads folder to Google Drive's "Reading Material" folder
gfile_dict = {}
for subdir, dirs, files in os.walk(downloads_folder):
    for src_file in files:
        if src_file.endswith('.mp3'):
            src_path = os.path.join(subdir, src_file)
            gfile_dict[src_file] = drive.CreateFile({'title':src_file, 'mimeType':'audio/mp3',
                                                     'parents': [{'kind': 'drive#fileLink',
                                                                  'id': tgt_folder_id}]})

            # Read mp3 file and set it as a content of this instance.
            gfile_dict[src_file].SetContentFile(src_path)
            
            # Upload the file
            try:
                gfile_dict[src_file].Upload()
                print('Uploaded %s (%s)' % (gfile_dict[src_file]['title'],
                                            gfile_dict[src_file]['mimeType']))
            except Exception as e:
                print('Upload failed for %s (%s): %s' % (gfile_dict[src_file]['title'],
                                                         gfile_dict[src_file]['mimeType'], e))
print('Upload completed.')


import os

# Trash all mp3s from Google Drive's "Reading Material" folder
for src_file in gfile_dict.keys():

    # Trash mp3 file
    try:
        gfile_dict[src_file].Trash()
        print('Trashed %s (%s)' % (gfile_dict[src_file]['title'],
                                   gfile_dict[src_file]['mimeType']))
    except Exception as e:
        print('Trash failed for %s (%s): %s' % (gfile_dict[src_file]['title'],
                                                gfile_dict[src_file]['mimeType'], e))
    
print('Trashing completed.')


import os

# Delete all mp3s in the Downloads folder
for src_file in gfile_dict.keys():
    src_path = os.path.join(downloads_folder, src_file)

    # Delete the file
    try:
        os.remove(src_path)
        print('Deleted %s (%s)' % (src_file,
                                   gfile_dict[src_file]['mimeType']))
    except Exception as e:
        print('Failed to delete %s (%s): %s' % (src_file,
                                                gfile_dict[src_file]['mimeType'], e))
print('Deleting completed.')



