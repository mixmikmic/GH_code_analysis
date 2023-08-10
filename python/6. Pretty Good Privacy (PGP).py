#Use GCM
def enc_key(public_key, message):
    pass

def dec_key(private_key, ciphertext):
    pass

def enc_msg(key, iv, msg):
    pass

def dec_msg(key, iv, ciphertext):
    pass

k1 = os.urandom(16)
iv = os.urandom(16)
msg = b"PyCon 2017 Crypto!!"

cipher = enc_msg(k1, iv, msg)
encrypted_key = enc_key(public_key, k1)

decrypted_key = dec_key(private_key, encrypted_key)
plaintext = dec_msg(decrypted_key, iv, cipher)

