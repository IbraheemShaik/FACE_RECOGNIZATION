'''# app.py
from flask import Flask, render_template, request, redirect, url_for
import threading

# Import the necessary functions from other scripts
from capture_images import capture_images
from model import train_model
from recognize import recognize_face

app = Flask(__name__)
# Home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form['name']
        capture_thread = threading.Thread(target=capture_images, args=(name,))
        capture_thread.start()
        return redirect(url_for('index'))
    return render_template('index.html')

# Train the model
@app.route('/train', methods=['POST'])
def train():
    
    train_thread = threading.Thread(target=train_model)
    train_thread.start()
    return redirect(url_for('index'))

# Recognize faces
@app.route('/recognize', methods=['POST'])
def recognize():
    recognize_thread = threading.Thread(target=recognize_face)
    recognize_thread.start()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)'''

from flask import Flask, render_template, request, redirect, url_for, jsonify
import threading
from capture_images import capture_images
from model import train_model
from recognize import recognize_face

app = Flask(__name__)

# Home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Add new person - Capture images for a new person
@app.route('/add_person', methods=['POST'])
def add_person():
    name = request.form['name']
    if not name:
        return jsonify({"error": "Name is required"}), 400
    capture_thread = threading.Thread(target=capture_images, args=(name,))
    capture_thread.start()
    return redirect(url_for('index'))

# Train the model with new and old data
@app.route('/train', methods=['POST'])
def train():
    train_thread = threading.Thread(target=train_model)
    train_thread.start()
    return redirect(url_for('index'))

# Recognize faces - Test live predictions
@app.route('/recognize', methods=['POST'])
def recognize():
    recognize_thread = threading.Thread(target=recognize_face)
    recognize_thread.start()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

