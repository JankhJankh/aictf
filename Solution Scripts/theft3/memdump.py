import pickle
#Head "\x80\x03\x63\x6b\x65\x72\x61\x73\x2e\x65\x6e\x67\x69\x6e\x65\x2e\x74\x72\x61\x69"
#Tail "\x03\x00\x00\x28\x68\x03\x88\x68\x6f\x5d\x72\xa8\x03\x00\x00\x75\x75\x75\x62\x2e"
head = b"\x80\x03\x63\x6b\x65"
tail = b"\x75\x75\x75\x62\x2e"
piece_size = 10 # 4 KiB

startpoint = 0
endpoint = 0
with open("./python.DMP", "rb") as in_file:
   piece = in_file.read()

for i in range(len(piece)):
    j = 0
    while piece[i+j] == head[j]:
        if(j == 4):
            startpoint = i
            print(startpoint)
            break
        j=j+1
    j = 0
    while piece[i+j] == tail[j]:
        if(j == 4):
            endpoint = i+j
            print(endpoint)
            break
        j=j+1

plaintext = piece[startpoint:endpoint+1]
model = pickle.loads(plaintext)
model.save('./model.h5')
