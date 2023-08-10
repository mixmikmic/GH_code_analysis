#One of the requirement you may have to transfer your files to/from from Bluemix Spark
#Service notebook enivornment to your enivornment.
#we will use pexpect module to achieve this.

#Bluemix spark instances typically have pexpect but if you get error from below
#statement, '!pip install --user pexpect' to install it.

import pexpect

#In above command we copy/download the file into scpsol directory under your tenant
#directory on your spark service instance.

#do '!mkdir scpsol' give any name to your directory but change it below

try:
       var_password  = "xxxxxxxxxx"
       var_command = "scp -o 'StrictHostKeyChecking no' your_username@remotehost.edu:/some/remote/directory/testfoobar.txt scpsol/"
       #make sure in the above command that username and hostname are according to your server or use ip if you don't have hostname.
       var_child = pexpect.spawn(var_command)
       i = var_child.expect(["password:", pexpect.EOF])

       if i==0: # send password                
               var_child.sendline(var_password)
               var_child.expect(pexpect.EOF)
       elif i==1: 
               print "Key or connection timeout"
               pass

except Exception as e:
       print "Something is not correct"
       print e

#!ls scpsol --if you want to see the downloaded file

#TO copy/upload your files from spark service tenant to your remote server.
#Use below command.Below command copies foobar.txt to remotehost.edu.

get_ipython().system('touch foobar.txt')

try:
       var_password  = "xxxxxxxxxxxx"
       var_command = "scp -o 'StrictHostKeyChecking no' foobar.txt your_username@remotehost.edu:/some/remote/directory"
       #make sure in the above command that username and hostname are according to your server
       var_child = pexpect.spawn(var_command)
       i = var_child.expect(["password:", pexpect.EOF])

       if i==0: # send password                
               var_child.sendline(var_password)
               var_child.expect(pexpect.EOF)
       elif i==1: 
               print "Key or connection timeout"
               pass

except Exception as e:
       print "Something is not correct"
       print e

#For references on scp use:- http://www.hypexr.org/linux_scp_help.php
#Credit
#http://stackoverflow.com/questions/22237486/scp-to-a-remote-server-using-pexpect

