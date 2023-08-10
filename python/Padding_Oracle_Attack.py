# Sources
# https://en.wikipedia.org/wiki/Padding_(cryptography)#PKCS7
# https://programmerin.blogspot.com/2011/08/python-padding-with-pkcs7.html

def PKCS7validate(input_string):
    d = list(input_string)
    if d[-1] == '\x01':
        return True
    for x in range(ord(d[-1])):
        if x == 0:
            pass
        else:
            if d[-1] == d[(x+1)*-1]:
                pass
            else:
                return False
    return True

print PKCS7validate("ICE ICE BABY\x04\x04\x04\x04") # Should print True
print PKCS7validate("ICE ICE BABY\x05\x05\x05\x05") # False
print PKCS7validate("ICE ICE BABY\x01\x02\x03\x04") # False
print PKCS7validate("ICE ICE BABY\x01\x02\x03\x01") # True - basis of attack

import urllib2
import binascii
cipher = "ecf5f6d6405e2ad74254ff211635e390"
url = "https://id0-rsa.pub/problem/cbc_padding_oracle/"
iv ="c6574d8a54c952a7f298673ee7063c16"
# Break the original iv into an array of 2 byte strings eg. ["c6","57"....]
chunks, chunk_size = len(iv), len(iv)/16
ogIVarray = [iv[i:i+chunk_size] for i in range(0, chunks, chunk_size) ]

    
def searchBytes(ivarray, x):
    '''Attempts everything from 00 -> ff for the byte I am testing'''
    print "Attempting to break byte: "+ str(x)
    for y in range(256):
        try:
            if y == 0:
                #when y ==0 dont increment our guess number, so that we guess 00
                ivGuess = "".join(ivarray)
                urllib2.urlopen(url+ivGuess+cipher).read()
            else:
                # get the part of the array we are brute forcing and increment
                ivGuess = int(ivarray[x], 16)
                ivGuess = hex(ivGuess + 1)
                ivGuess = ivGuess[2:]
                if len(ivGuess) == 1:
                    ivGuess ="0"+ivGuess
                ivarray[x] = ivGuess
                ivGuess = "".join(ivarray)
                urllib2.urlopen(url + ivGuess + cipher).read()
        except urllib2.HTTPError as er:
            #the oracle said no this padding doesnt work try again
            pass
        else:
            if x == 0:
                if ivarray == ogIVarray:
                     # if it is equal to the original string ignore and continue
                    print "Original attempt"
                    pass
                else:
                #you found the solution compute and add the intermediate value to s, and contune on to the next bytes
                    print "got it "+ "".join(ivarray)
                    return ivarray[x]
            print "got it "+ "".join(ivarray)
            return ivarray[x]

def setupIv(ivg, count, s):
    '''sets up our guessing iv by using the known s "intermediate" values to compute the padding value we want'''
    #s is our intermediate value, we need to ensude
    im = []
    for z in range(15,15-(count-1),-1):
        bae = hex(int(s[15-z],16) ^ count)[2:]
        if len(bae) == 1:
            bae="0"+bae
        im.append( bae)
    im = im[::-1]
    cc =0
    for x in im:
        ivg[15-cc]= x
        cc=cc+1
    return ivg

def breakcbc():
    #s is my intermediate value, aka what I am solving for
    #Once solved it = 820504c41f8604e6becc2e70a2053f15
    s = []

    # this will be my iv guesser, so just an array that I will modify bite by bite from right to left.
    chunks, chunk_size = len(iv), len(iv)/16
    ivarray = [iv[i:i+chunk_size] for i in range(0, chunks, chunk_size) ]

    count = 1 #keeps track of which padding im guessing 1= \x01 2=\x02\x02... etc
    for x in range(15,-1,-1):
        if x != 15:
            ivarray = setupIv(ivarray, count, s)
        ivarray[x] = "00"
        #print "".join(ivarray)
        val = searchBytes(ivarray,x)
        print "Byte that broke it: "+ val
        i = hex(int(val,16)^ count)[2:]
        if len(i) == 1:
            i = "0"+i
        s = [i] + s
        #print "".join(s)
        count = count + 1
    ans = hex(int("".join(s),16)^int(iv,16))
    print binascii.unhexlify(ans[2:-1])

# Once solved intermediatevalue = 820504c41f8604e6becc2e70a2053f15
#SOLUTION = Drinkovaltine
breakcbc()



