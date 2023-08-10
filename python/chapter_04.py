import io

def caesar_cipher(plaintext, shift):
    ciphertext = [ chr((ord(c) - ord('A') + shift) % 26 + ord('A')) 
                  for c in plaintext ]
    return ''.join(ciphertext)
    
ciphertext = caesar_cipher('IAMSEATEDINANOFFICE', 5)
print(ciphertext)

print(caesar_cipher(ciphertext, -5))

def ord0(c):
    return ord(c) - ord('A')

def otpad_encrypt(plaintext, randomtext):
    encrypted = [ chr((ord0(c) + ord0(r)) % 26 + ord('A')) 
                 for c, r in zip(plaintext, randomtext) ]
    return ''.join(encrypted)

plaintext = "IFYOUREALLYWANTTO"
randomtext = "RDUUWJEMCJJXZDOWJ"

ciphertext = otpad_encrypt(plaintext, randomtext)
print(ciphertext)

def otpad_decrypt(ciphertext, randomtext):
    decrypted = [ chr((ord0(c) - ord0(r)) % 26 + ord('A')) 
                 for c, r in zip(ciphertext, randomtext) ]
    return ''.join(decrypted)

plaintext = otpad_decrypt(ciphertext, randomtext)
print(plaintext)

def otpad_red_herring(plaintext, ciphertext):
    red_herring = [ chr((ord0(c) - ord0(p)) % 26 + ord('A'))
                   for p, c in zip(plaintext, ciphertext) ]
    return ''.join(red_herring)

red_herring = otpad_red_herring("IAMSEATEDINANOFFI", ciphertext)
print(red_herring)

print(otpad_decrypt(ciphertext, red_herring))

red_herring = otpad_red_herring("RENDEZVOUSATNIGHT", ciphertext)
print(red_herring)

print(otpad_decrypt(ciphertext, red_herring))

def otpad_xencrypt(plaintext, randomtext):
    encrypted = [ chr(ord(p) ^ ord(r)) for p, r in zip(plaintext, randomtext) ]
    return ''.join(encrypted)

ciphertext = otpad_xencrypt(plaintext, randomtext)
print(ciphertext)

print(ciphertext.encode('utf8'))

decrypted = otpad_xencrypt(ciphertext, randomtext)

print(decrypted)

g = 2
p = 13

for x in range(1, 13):
    print(pow(g, x), pow(g, x, p))

p = 0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AAAC42DAD33170D04507A33A85521ABDF1CBA64ECFB850458DBEF0A8AEA71575D060C7DB3970F85A6E1E4C7ABF5AE8CDB0933D71E8C94E04A25619DCEE3D2261AD2EE6BF12FFA06D98A0864D87602733EC86A64521F2B18177B200CBBE117577A615D6C770988C0BAD946E208E24FA074E5AB3143DB5BFCE0FD108E4B82D120A92108011A723C12A787E6D788719A10BDBA5B2699C327186AF4E23C1A946834B6150BDA2583E9CA2AD44CE8DBBBC2DB04DE8EF92E8EFC141FBECAA6287C59474E6BC05D99B2964FA090C3A2233BA186515BE7ED1F612970CEE2D7AFB81BDD762170481CD0069127D5B05AA993B4EA988D8FDDC186FFB7DC90A6C08F4DF435C934063199FFFFFFFFFFFFFFFF

g = 2

import random

a = random.getrandbits(256)
print(a)

alice_to_bob = pow(g, a, p)
print(alice_to_bob)

p = 0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AAAC42DAD33170D04507A33A85521ABDF1CBA64ECFB850458DBEF0A8AEA71575D060C7DB3970F85A6E1E4C7ABF5AE8CDB0933D71E8C94E04A25619DCEE3D2261AD2EE6BF12FFA06D98A0864D87602733EC86A64521F2B18177B200CBBE117577A615D6C770988C0BAD946E208E24FA074E5AB3143DB5BFCE0FD108E4B82D120A92108011A723C12A787E6D788719A10BDBA5B2699C327186AF4E23C1A946834B6150BDA2583E9CA2AD44CE8DBBBC2DB04DE8EF92E8EFC141FBECAA6287C59474E6BC05D99B2964FA090C3A2233BA186515BE7ED1F612970CEE2D7AFB81BDD762170481CD0069127D5B05AA993B4EA988D8FDDC186FFB7DC90A6C08F4DF435C934063199FFFFFFFFFFFFFFFF 

g = 2

b = random.getrandbits(256)
print('b =', b)

bob_to_alice = pow(g, b, p)
print(bob_to_alice)

shared_secret_alice = pow(bob_to_alice, a, p)
print(shared_secret_alice)

shared_secret_bob = pow(alice_to_bob, b, p)
print(shared_secret_bob)

shared_secret_alice == shared_secret_bob

def exponentiation(g, x):
    print('{0:>10} {1:>10} {2:>10}'.format('c', 'r', 'd'))
    c = g
    d = x
    r = 1
    while d > 0:
        print(f'{c:-10d} {r:-10d} {bin(d):>10}')
        if d & 0b1 == 1:
            r = r * c
        d = d >> 1
        c = c * c
    return r

exponentiation(13, 13)

def mod_exponentiation(g, x, p):
    print('{0:>10} {1:>10} {2:>10}'.format('c', 'r', 'd'))
    c = g % p
    d = x
    r = 1
    while d > 0:
        print(f'{c:-10d} {r:-10d} {bin(d):>10}')
        if d & 0b1 == 1:
            r = (r * c) % p
        d = d >> 1
        c = (c * c) % p
    return r

mod_exponentiation(155, 235, 391)

