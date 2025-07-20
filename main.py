import os
import io
import re
import base64
import sqlite3
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf

# Flask app setup
app = Flask(__name__,
            template_folder='src/templates',
            static_folder='src/static')

# Create folders if not exist
os.makedirs('database', exist_ok=True)
os.makedirs('face_data', exist_ok=True)

# Load TFLite model once
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ Load labels (split to get only internId like 'thy76', not '0 thy76')
with open("labels.txt", "r") as f:
    labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]

# Image preprocessing
def preprocess(image):
    image = image.resize((224, 224))  # match your model input size
    image = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

# Prediction function
def predict(image):
    input_data = preprocess(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    top_index = np.argmax(output_data)
    print(f"🔍 Predicted label: {labels[top_index]}, Confidence: {output_data[top_index]}")
    return labels[top_index], float(output_data[top_index])

# Routes
@app.route('/')
def login():
    return render_template('intern_login.html')

@app.route('/register')
def register():
    return render_template('intern_register.html')
@app.route('/intern')
def dashboard():
    return render_template('intern.html')

@app.route('/submit_registration', methods=['POST'])
def submit_registration():
    data = request.json
    intern_id = data['internId']
    username = data['username']
    password = data['password']
    images = data['images']  # list of base64 images

    # Save user to database
    conn = sqlite3.connect('database/interns.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS interns (intern_id TEXT, username TEXT, password TEXT)')
    c.execute('INSERT INTO interns (intern_id, username, password) VALUES (?, ?, ?)',
              (intern_id, username, password))
    conn.commit()
    conn.close()

    # Save images
    folder = os.path.join('face_data', intern_id)
    os.makedirs(folder, exist_ok=True)

    for i, img_base64 in enumerate(images):
        img_data = img_base64.split(',')[1]
        with open(os.path.join(folder, f'face_{i+1}.png'), 'wb') as f:
            f.write(base64.b64decode(img_data))

    return jsonify({'message': 'Registration successful'})

@app.route('/login_face', methods=['POST'])
def login_face():
    data = request.json
    intern_id = data['internId']
    username = data['username']
    password = data['password']
    image_data = data['image']

    # Check credentials
    conn = sqlite3.connect('database/interns.db')
    c = conn.cursor()
    c.execute("SELECT * FROM interns WHERE intern_id=? AND username=? AND password=?", (intern_id, username, password))
    result = c.fetchone()
    conn.close()

    if not result:
        return jsonify({'status': 'error', 'message': 'Invalid credentials'})

    # Decode image
    try:
        img_str = re.search(r'base64,(.*)', image_data).group(1)
        img_bytes = base64.b64decode(img_str)
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        print("Image decode error:", e)
        return jsonify({'status': 'error', 'message': 'Failed to process image'})

    # Predict
    print("🚀 Reached before prediction")
    predicted_label, confidence = predict(image)
    print(f"🎯 Predict: {predicted_label}, confidence: {confidence}")

    if predicted_label == intern_id and confidence > 0.80:
        return jsonify({'status': 'success', 'message': 'Login successful!'})
    else:
        return jsonify({'status': 'error', 'message': 'Face does not match.'})

if __name__ == '__main__':
    app.run(debug=True)
