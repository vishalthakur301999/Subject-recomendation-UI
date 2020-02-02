import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template
import pickle
import model
app = Flask(__name__) #Initialize the flask App
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    model = pickle.load(open('model.pkl', 'rb'))
    features = request.form['experience']
    features2 = int(request.form['score'])
    prediction = model(features, features2)
    prediction_text = prediction.to_json(orient='split')
    return render_template('index.html', **locals())

if __name__ == "__main__":
    app.run(debug=True)