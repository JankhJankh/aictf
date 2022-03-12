import requests

requests.get("http://127.0.0.1:5000/resetpoison")
requests.get("http://127.0.0.1:5000/rebuildpoison")

for i in range(0,20):
  r1 = requests.get("http://127.0.0.1:5000/addpoison?data=BEES,1")

requests.get("http://127.0.0.1:5000/rebuildpoison")
r2 = requests.get("http://127.0.0.1:5000/checkpoison")
print(r2.text)
