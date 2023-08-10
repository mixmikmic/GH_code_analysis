#Importing Dependencies

from Crypto import Random
from Crypto.Cipher import AES
import base64
import hashlib

#copying plaintext from file
PlainTextFile=open("PlainText.txt")
plaintext=PlainTextFile.read()
plaintext

#AES Example

BS=16
pad = lambda s: s + (BS - len(s) % BS) * chr(BS - len(s) % BS) 
unpad = lambda s : s[:-ord(s[len(s)-1:])]

def encrypt(raw,key):
    CipherText=""
    CipherTextFile=open("CipherText.txt",'w')
    raw = pad(raw)
    iv = Random.new().read( AES.block_size )
    cipher = AES.new(key, AES.MODE_CBC,iv )
    CipherText=base64.b64encode( iv + cipher.encrypt( raw ) )
    CipherTextFile.write(CipherText)
    return CipherText

def decrypt(enc,key):
    PlainText=""
    PlainTextFile=open("PlainText.txt",'w')
    enc = base64.b64decode(enc)
    iv = enc[:16]
    cipher = AES.new(key, AES.MODE_CBC,iv)
    PlainText=unpad(cipher.decrypt( enc[16:] ))
    PlainTextFile.write(PlainText)
    return PlainText

#Main Block

#acts as the key input by user
passphrase='avhirup'
#key should be 16 bit,so generating 16 byte hash value using passphrase entered by user 
key=hashlib.sha256(passphrase).digest()
cipher=encrypt(plaintext,key)

print decrypt(cipher,key)





