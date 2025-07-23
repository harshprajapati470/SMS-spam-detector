from flask import Flask, request, jsonify, send_from_directory
import joblib
import string
import nltk
from nltk.corpus import stopwords
import os

nltk.download('stopwords')

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    stop_words = stopwords.words('english')
    return ' '.join([word for word in words if word not in stop_words])

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    msg = data.get('message')
    if not msg:
        return jsonify({'error': 'No message provided'}), 400

    cleaned = clean_text(msg)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)[0]
    return jsonify({'prediction': 'Spam' if pred == 1 else 'Ham'})

if __name__ == '__main__':
    app.run(debug=True)
