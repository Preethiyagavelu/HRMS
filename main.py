# main.py
import os
import io
import re
import base64
import sqlite3
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf
from pyngrok import ngrok
import time

app = Flask(__name__, template_folder='src/templates', static_folder='src/static')

os.makedirs('database', exist_ok=True)
os.makedirs('face_data', exist_ok=True)

interpreter = tf.lite.Interpreter(model_path="face_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("label.txt", "r") as f:
    labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]

def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

def predict(image):
    input_data = preprocess(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    top_index = np.argmax(output_data)
    return labels[top_index], float(output_data[top_index])

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
    images = data['images']

    conn = sqlite3.connect('database/interns.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS interns (intern_id TEXT, username TEXT, password TEXT)')
    c.execute('INSERT INTO interns (intern_id, username, password) VALUES (?, ?, ?)', (intern_id, username, password))
    conn.commit()
    conn.close()

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

    conn = sqlite3.connect('database/interns.db')
    c = conn.cursor()
    c.execute("SELECT * FROM interns WHERE intern_id=? AND username=? AND password=?", (intern_id, username, password))
    result = c.fetchone()
    conn.close()

    if not result:
        return jsonify({'status': 'error', 'message': 'Invalid credentials'})

    try:
        img_str = re.search(r'base64,(.*)', image_data).group(1)
        img_bytes = base64.b64decode(img_str)
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({'status': 'error', 'message': 'Failed to process image'})

    predicted_label, confidence = predict(image)

    if predicted_label == intern_id and confidence > 0.80:
        punch_in = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        conn = sqlite3.connect('database/interns.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS attendance 
                     (intern_id TEXT, date TEXT, punch_in TEXT, punch_out TEXT, duration TEXT)''')
        c.execute("INSERT INTO attendance (intern_id, date, punch_in) VALUES (?, ?, ?)",
                  (intern_id, punch_in.split()[0], punch_in))
        conn.commit()
        conn.close()

        return jsonify({'status': 'success', 'message': 'Login successful!', 'punchInTime': punch_in, 'internId': intern_id})
    else:
        return jsonify({'status': 'error', 'message': 'Face does not match.'})

@app.route('/submit_punchout', methods=['POST'])
def submit_punchout():
    data = request.get_json(force=True)
    intern_id = data['internId']
    punch_out = data['punchOutTime']
    duration = data['duration']
    today = punch_out.split()[0]

    conn = sqlite3.connect('database/interns.db')
    c = conn.cursor()

    for _ in range(3):
        conn.commit()
        c.execute("""
            SELECT rowid FROM attendance
            WHERE intern_id = ? AND date = ? AND punch_out IS NULL
            ORDER BY punch_in DESC LIMIT 1
        """, (intern_id, today))
        result = c.fetchone()
        if result:
            break
        time.sleep(0.2)

    if result:
        rowid = result[0]
        c.execute("""
            UPDATE attendance SET punch_out = ?, duration = ?
            WHERE rowid = ?
        """, (punch_out, duration, rowid))
        conn.commit()
        conn.close()
        return jsonify({'status': 'success', 'message': 'Punch-out time saved successfully.'})
    else:
        conn.close()
        return jsonify({'status': 'error', 'message': 'No punch-in record found to update.'})

if __name__ == '__main__':
    ngrok.set_auth_token("2zPJ6U17aZxGZP9wBd6YNaIgQbs_4ZytQ8C4dSKS7s9tGpK5Q")
    ngrok_tunnel = ngrok.connect(5000)
    print(' * Public URL:', ngrok_tunnel.public_url)
    app.run(debug=False)
