from flask import Flask, render_template, request
from transformers import AutoTokenizer, TFDistilBertModel
import numpy as np
from keras.models import load_model
from keras.saving import register_keras_serializable

app = Flask(__name__)

# Load model BERT dari HuggingFace
bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# Daftarkan custom layer agar bisa dimuat saat load_model
@register_keras_serializable()
def call_bert(inputs):
    input_ids, attention_mask = inputs
    outputs = bert_model([input_ids, attention_mask])[0]  # ambil hidden states
    return outputs[:, 0, :]  # ambil token [CLS]

# Load tokenizer dan model terlatih
tokenizer = AutoTokenizer.from_pretrained('./tokenizers')
model = load_model('./models/best_model.keras', compile=False, custom_objects={'call_bert': call_bert})

# Route halaman utama
@app.route('/')
def home():
    return render_template('index.html')

# Route untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    
    # Tokenisasi input URL
    tokens = tokenizer(
        url,
        padding='max_length',
        max_length=38,
        truncation=True,
        return_tensors='tf'
    )
    
    # Prediksi dengan model
    prediction = model.predict([tokens['input_ids'], tokens['attention_mask']])
    phishing_prob = float(prediction[0][0]) * 100
    legitimate_prob = 100 - phishing_prob

    # Determine color and classification based on probabilities
    CONFIDENCE_THRESHOLD = 65.0
    
    if legitimate_prob < CONFIDENCE_THRESHOLD and phishing_prob < CONFIDENCE_THRESHOLD:
        result_class = "alert-warning"
        classification = "SUSPICIOUS"
    elif legitimate_prob > phishing_prob:
        result_class = "alert-success"
        classification = "LEGITIMATE"
    else:
        result_class = "alert-danger"
        classification = "PHISHING"

    return render_template('index.html', 
                         url=url, 
                         result=classification, 
                         result_class=result_class,
                         legitimate_prob=legitimate_prob,
                         phishing_prob=phishing_prob)

if __name__ == '__main__':
    app.run(debug=True)
