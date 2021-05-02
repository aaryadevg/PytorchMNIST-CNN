from flask import Flask, request, jsonify
from torch_utils import TransformImg, GetPrediction

app = Flask(__name__)


def AllowedFile(fname: str):
    ALLOWED_EXT = set(("jpeg", "jpg", "png"))
    if '.' in fname:
        ext : str = fname.rsplit('.', 1)[1]
        #print(fname, ext)
        ext = ext.lower()
        if ext in ALLOWED_EXT:
            return True
    
    return False

@app.route("/predict", methods = ["POST"])
def predict():
    if request.method == "POST":
        file = request.files.get("Image")
        
        if file is None or file.filename == "":
            return jsonify({"ERROR" : "No input file"})
        
        if not AllowedFile(file.filename):
            return jsonify({"ERROR" : "File format not supported"})
        
        try:
            ImgBytes  = file.read()
            ImgTensor = TransformImg(ImgBytes)
            Predicted = GetPrediction(ImgTensor).item()
            data = {
                "Prediction"     : Predicted,
                "PredictedClass" : str(Predicted)
            }
            return jsonify(data)
            
        except:
           return jsonify({"ERROR" : "Error during prediction"})
    else:
        return jsonify({"ERROR" : "Incorrect request method"})
        