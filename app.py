from flask import Flask, render_template, request, redirect, url_for, session, g
import sqlite3
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Add a secret key for session management

DATABASE = 'users.db'

# Load the models and encoders
with open('Scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('PCA.pkl', 'rb') as file:
    pca = pickle.load(file)
with open('LabelEncoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)
with open('Voting.pkl', 'rb') as file:
    model = pickle.load(file)

def predict_new_data(data, scaler, pca, model, label_encoder):
    # Scale the data
    scaled_data = scaler.transform(data)
    # Apply PCA
    pca_data = pca.transform(scaled_data)
    # Predict the encoded label
    encoded_prediction = model.predict(pca_data)
    # Convert the encoded label back to the original categorical label
    categorical_prediction = label_encoder.inverse_transform(encoded_prediction)
    return categorical_prediction[0]

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = cursor.fetchone()
        if user:
            session['username'] = username
            return redirect(url_for('predict'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db()
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('register.html', error='Username already exists')
    return render_template('register.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            # Get data from form
            data = [float(request.form.get(x)) for x in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
            feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            data_df = pd.DataFrame([data], columns=feature_names)
            # Make prediction
            prediction = predict_new_data(data_df, scaler, pca, model, label_encoder)
            # Generate image filename (replace with actual logic)
            prediction_image = f"{prediction}.jpg"
            return render_template('predict.html', prediction_text=f'Predicted Crop: {prediction}', prediction_image=prediction_image)
        except Exception as e:
            error_message = f"Prediction error: {str(e)}"
            return render_template('predict.html', error_message=error_message)
    else:
        return render_template('predict.html')

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
