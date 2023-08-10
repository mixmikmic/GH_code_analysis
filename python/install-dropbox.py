cd

get_ipython().system('wget -O dropbox.tar.gz "http://www.dropbox.com/download/?plat=lnx.x86_64"')

get_ipython().system('tar -xzf dropbox.tar.gz')

get_ipython().system('~/.dropbox-dist/dropboxd')

# finally: reboot!
get_ipython().system('/sbin/shutdown -r now')

