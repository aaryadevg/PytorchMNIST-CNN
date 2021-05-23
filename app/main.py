from flask import Flask, request, jsonify
from torch_utils import TransformImg, GetPrediction

app = Flask(__name__)

# Checks if file has correct extension
def FileIsAllowed(fileName: str):
    ALLOWED_EXT = set(("jpeg", "jpg", "png"))
    if '.' in fileName:
        fileExtension : str = fileName.rsplit('.', 1)[1]
        fileExtension = fileExtension.lower()
        if fileExtension in ALLOWED_EXT:
            return True
    
    return False

@app.route("/predict", methods = ["POST"])
def predict():
    if request.method == "POST":
        file = request.files.get("Image")
        
        if file is None or file.filename == "":
            return jsonify({"ERROR" : "No input file"})
        
        if not FileIsAllowed(file.filename):
            return jsonify({"ERROR" : "File format not supported"})
        
        # Not sure if it's a good idea to return exception here
        # because an attacker can figure the libraries used and search for exploits within the libraries?
        # Leaving it here for debug purposes
        try:
            ImgBytes  = file.read()
            ImgTensor = TransformImg(ImgBytes)
            Predicted = GetPrediction(ImgTensor).item()
            data = {
                "Prediction" : Predicted
            }
            return jsonify(data)        
        except (Exception) as ex:
           return jsonify({"ERROR" : "Error during prediction",
                           "EXCEPTION" : ex.__str__()}) 
    else:
        return jsonify({"ERROR" : "Incorrect request method"}) # I think flask handles to check if it was incorrect request type
        