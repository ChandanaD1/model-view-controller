# model view controller

from flask import Flask, jsonify, request
from hw29part1 import get_prediction

app = Flask(__name__)
@app.route("/predict-alphabet",methods=["POST"])

def predict_data():
    image = request.files.get("alphabet")
    prediction = get_prediction(image)
    return(jsonify({
        "predict":prediction
    }),200)

if __name__=="__main__":
    app.run()
