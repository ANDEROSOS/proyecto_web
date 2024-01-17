# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:25:22 2024

@author: USUARIO
"""
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Cargar el modelo de TensorFlow
model = tf.keras.models.load_model('modelo_red_neuronal.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        snf = float(request.form['snf'])
        awm = float(request.form['awm'])
        lact = float(request.form['lact'])
        fat = float(request.form['fat'])
        protein = float(request.form['protein'])
        
        # Realizar la predicción
        inputs = np.array([[snf, awm, lact, fat, protein]], dtype=np.float32)
        prediction = model.predict(inputs)
        result = prediction[0][0] * 100

        return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

