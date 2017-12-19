from flask import Flask, jsonify, request
from sklearn.externals import joblib
import pandas as pd
import pickle
import numpy as np


app = Flask(__name__)

clf = joblib.load('wiml_model.pkl')



@app.route('/predict_api', methods=['POST'])
def predict():
     # Error checking
     data = request.get_json(force=True)

     # Convert JSON to numpy array
     predict_request = [data['SepalLengthCm'],data['SepalWidthCm'],data['PetalLengthCm'],data['PetalWidthCm']]
     predict_request = np.array([predict_request])

     # Predict using the random forest model
     y = clf.predict(predict_request).tolist()

     # Return prediction
     output = [y[0]]
     return jsonify(results=output)

if __name__ == '__main__':
     app.run(port = 9000, debug = True)