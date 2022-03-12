import requests

r2 = requests.get("http://127.0.0.1:5000/checktoken?line1=493&line2=337")
print(r2.text)

