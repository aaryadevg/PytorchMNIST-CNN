import requests

URL = "http://127.0.0.1:5000/predict"
resp = requests.post(URL, files={"Image" : open("OtherSize.png", "rb")})
print(resp.text)