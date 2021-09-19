import requests

# TODO: Check prediction with all files
# TODO: Put PyTest maybe?
TEST_FILE = "OtherSize.png"
URL = "http://127.0.0.1:5000/predict"
resp = requests.post(URL, files={"Image" : open(TEST_FILE, "rb")})
print(resp.text)
