#Importing Dependencies

from Crypto import Random
from Crypto.Cipher import DES
import base64
import hashlib

#copying plaintext from file
PlainTextFile=open("PlainText.txt")
plaintext=PlainTextFile.read()
plaintext

#DES Example
#BS is BLOCK SIZE
#Padding and Unpadding required when input size is not a multiple of BS

BS=8
pad = lambda s: s + (BS - len(s) % BS) * chr(BS - len(s) % BS) 
unpad = lambda s : s[:-ord(s[len(s)-1:])]

def encrypt(raw,key):
    CipherText=""
    CipherTextFile=open("CipherText.txt",'w')
    raw = pad(raw)
    cipher = DES.new(key, DES.MODE_ECB)
    CipherText=cipher.encrypt( raw ) 
    CipherTextFile.write(CipherText)
    return CipherText

def decrypt(enc,key):
    PlainText=""
    PlainTextFile=open("PlainText.txt",'w')
    cipher = DES.new(key, DES.MODE_ECB )
    PlainText=unpad(cipher.decrypt( enc[:8] ))
    PlainTextFile.write(PlainText)
    return PlainText

#Main Block
#acts as the key input by user
key='01234567'
cipher=encrypt(plaintext,key)

decrypt(cipher,key)





