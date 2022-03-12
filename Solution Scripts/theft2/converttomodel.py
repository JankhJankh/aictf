import pickle
from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes
from base64 import b64encode
with open("rockyou-20.txt") as f:
    content = f.read().splitlines()

f = open("./encpickle", "rb")
daytas = f.read()
    
for line in content:
    key = bytes((line*16)[0:16], 'utf-8')
    nonce = bytes((line*16)[0:16], 'utf-8')
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    try:
        plaintext = cipher.decrypt(daytas)
        model = pickle.loads(plaintext)
        model.save('./model.h5')
    except:
        pass