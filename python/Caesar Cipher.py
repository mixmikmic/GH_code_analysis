#Reads File to Encrypt or Decrypt

PlainTextFile=open("CaesarCipherText.txt")
plaintext=PlainTextFile.read()
plaintext

#Encrypt Function
##input txt,key
##output cipherfile
def Encrypt(ptxt,key):
    CipherText=""
    CipherTextFile=open("CaesarCipherText.txt",'w')
    for ch in ptxt:
        if ch.isalpha():
            if ch.islower():
                ch=chr(((ord(ch)-ord('a')+key)%26)+ord('a'))
            if ch.isupper():
                ch=chr(((ord(ch)-ord('A')+key)%26)+ord('A'))
        CipherText=CipherText+ch
    CipherTextFile.write(CipherText)

#Encrypt Function
##input txt,key
##output plaintextfile
def Decrypt(ptxt,key):
    PlainText=""
    PlainTextFile=open("PlainText.txt",'w')
    for ch in ptxt:
        if ch.isalpha():
            if ch.islower():
                ch=chr(((ord(ch)-ord('a')-key)%26)+ord('a'))
            if ch.isupper():
                ch=chr(((ord(ch)-ord('A')-key)%26)+ord('A'))
        PlainText=PlainText+ch
    PlainTextFile.write(PlainText)

#Brute Force approach to find key

import enchant as en
import re
Endict = en.Dict("en_US")

def DecryptWords(wrd,key):
    PlainText=""
    for ch in wrd:
        if ch.isalpha():
            if ch.islower():
                ch=chr(((ord(ch)-ord('a')-key)%26)+ord('a'))
            if ch.isupper():
                ch=chr(((ord(ch)-ord('A')-key)%26)+ord('A'))
        PlainText=PlainText+ch
    return PlainText

def FindKey(ptxt):
    i=0;
    ptxt=re.split("\W+",ptxt)
    for key in range(27):
        res=True
        for wrd in ptxt:
            if wrd:
                tmpres=Endict.check(DecryptWords(wrd,key))
                res=tmpres and res
        if res:
            print 'key:',key
            return

Encrypt(plaintext,3)

FindKey(plaintext)

Decrypt(plaintext,3)



