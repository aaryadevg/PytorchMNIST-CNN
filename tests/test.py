import requests

# TODO: Check prediction with all files
# TODO: Put PyTest maybe?
URL = "http://127.0.0.1:5000/predict"
resp = requests.post(URL, files={"Image" : open("OtherSize.png", "rb")})
print(resp.text)