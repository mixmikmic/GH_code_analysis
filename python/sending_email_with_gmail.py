import smtplib
from email.mime.text import MIMEText
from email.utils import COMMASPACE
from getpass import getpass

GMAIL_USER = input("Enter gmail email: ")
GMAIL_PWD = getpass("Enter password: ")

SUBJECT = 'Email from myself'
BODY = 'You ready for some donuts?'
TO = 'jonalvarez624@gmail.com'

def sendEmail(sender, pwd, to, subject, message):
    recipient = to if type(to) is list else [to]
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = COMMASPACE.join(recipient)
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.ehlo()
    server.starttls()

    try:
        server.login(sender,pwd)
        print('Successfully authenticated...')
    except smtplib.SMTPAuthenticationError:               # Check for authentication error
        return " Authentication ERROR"

    try:
        server.sendmail(sender,recipient,msg.as_string())
        print('Email sent!')
    except smtplib.SMTPRecipientsRefused:                # Check if recipient's email was accepted by the server
        return "ERROR"
    server.quit()

sendEmail(GMAIL_USER, GMAIL_PWD, TO, SUBJECT, BODY)

