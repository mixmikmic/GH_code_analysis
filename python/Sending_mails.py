#Importing the necessary library

import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText 
from email.MIMEBase import MIMEBase
from email import encoders

""" Creating an object for manipulating the methods of the library 
    Method used -- .SMTP(  ,  )
    Parameters expected - hostname, the site where you want to login to 
                          port name, 587 (if using TLS ; 465 if using SSL)"""
    

sobj = smtplib.SMTP(host='smtp.gmail.com',port = 587)

#ssl is the default ____ in smtplib, however if tls is being used, the program needs to be notified seperately  

sobj.starttls()

#OPTIONAL -> We check whether the connection has been established successully, done using ehlo() method 
#uncomment the next line to see it work 
#sobj.ehlo()

""" The next step is to log in. 
The username can be give directly, however passwords should not be hardcoded in a program 
instead, provide the password, on input prompt whilst setting it to a variable. 
Then login using it. """

password = raw_input(" Enter Password - ")
username = ''
sobj.login(username,password)

"""Note - 1. The email service provider might block logins from low level applications
          This has to disabled in the account settings,(procedure depends on your email provider,no coding involved)
          For google - "allow less secure apps"
          2. Also in case of 2-page verification, the password to be entered here, has to be an application specific
          password, again to be set using the email provider. 
        In addition,after logging in from the browser go to the following page 
        http://www.google.com/accounts/DisplayUnlockCaptcha 
        and click continue. Account access will be enabled. """

#Specify the sender and reciever address, recievers can be a list as well 

fromaddr = ""
toaddr = ""

#Creating an object for the message contents 

msg = MIMEMultipart()

# Enter the message contents 

msg['From'] = fromaddr
msg['To'] = toaddr
msg['Subject'] = ""
body = " "

#Treating the body as a text object, we attach the body to the rest of the msg 

msg.attach(MIMEText(body,'plain'))

#For sending attachments, give the name of the file to be attached [***OPTIONAL***, uncomment the lines to make it work]
#filename = ''
#opening the attachment, to do so the path to the file needs to be specified.
#attachment = open('path',"rb")
##opening in reading and binary mode.
#part = MIMEBase('application','octet-stream')
#part.set_payload((attachment).read())
#encoders.encode_base64(part)
#part.add_header('Content Disposition',"attachment; filename = %s" %filename)
#msg.attach(part)

#Converting the message to a string, so that it can be sent. And then sending the mail 

text = msg.as_string()
sobj.sendmail(fromaaddr,toaddr,text)

#Closing the connection established safely 

sobj.quit()

