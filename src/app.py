import base64
import io
import os
import cv2 as cv
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image
from tensorflow.keras.models import load_model
from DisplayDisease import DisplayDisease

# ── Resolve paths relative to this file (works regardless of CWD) ────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create the Flask application
app = Flask(__name__)
model = None
multiModel = None
dt = DisplayDisease()


# ── Model loading ─────────────────────────────────────────────────────────────

def initialize_model():
    global model, multiModel
    binary_path = os.path.join(BASE_DIR, 'epoch10_sgd_acc96Point76.h5')
    multi_path  = os.path.join(BASE_DIR, 'multi-model-30K-epouch20.h5')
    print(f"Loading binary model from:     {binary_path}")
    print(f"Loading multiclass model from: {multi_path}")
    model      = load_model(binary_path)
    multiModel = load_model(multi_path)
    print("[OK] Both models loaded successfully.")


initialize_model()


# ── Image preprocessing ───────────────────────────────────────────────────────

def preprocess_binary_image(image: Image.Image):
    """Resize, normalise, and replicate into 3 inputs for the multi-head model."""
    image = image.convert('RGB').resize((128, 128))
    arr   = np.array(image, dtype=np.float32) / 255.0
    arr   = np.expand_dims(arr, axis=0)
    return [arr, arr, arr]


def preprocess_multiClass_image(image: Image.Image):
    """Resize, ensure RGB, normalise, and replicate into 3 inputs."""
    image = image.convert('RGB').resize((128, 128))
    arr   = np.array(image, dtype=np.float32)
    # OpenCV is not needed here – PIL already gives RGB
    arr   = arr / 255.0
    arr   = np.expand_dims(arr, axis=0)
    return [arr, arr, arr]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/binary.html', methods=['GET', 'POST'])
def binary():
    if request.method == 'POST':
        image_file = request.files.get('image')
        if not image_file:
            return jsonify({'error': 'No image uploaded'}), 400

        image = Image.open(image_file)
        processed = preprocess_binary_image(image)
        prediction = model.predict(processed)
        confidence = np.max(prediction)
        predicted_class = np.argmax(prediction)

        if confidence < 0.6:
            result = 'Uncertain (Low Confidence)'
        else:
            result = 'No Tumor Detected' if predicted_class == 0 else 'Tumor Detected'
            
        return jsonify({'result': result, 'confidence': float(confidence)})

    return render_template('binary.html')


@app.route('/multi.html', methods=['GET', 'POST'])
def multi():
    if request.method == 'POST':
        image_file = request.files.get('image')
        if not image_file:
            return jsonify({'error': 'No image uploaded'}), 400

        image = Image.open(image_file)
        # Strictly classify tumor type (never return "No Tumor")
        processed_multi = preprocess_multiClass_image(image)
        prediction = multiModel.predict(processed_multi)
        confidence = np.max(prediction)
        predicted_class = np.argmax(prediction)

        if confidence < 0.6:
            return jsonify({'result': 'Uncertain (Low Confidence)', 'confidence': float(confidence)})
            
        labels = {0: 'Glioma', 1: 'Meningioma', 2: 'Pituitary'}
        return jsonify({'result': labels.get(predicted_class, 'Unknown'), 'confidence': float(confidence)})

    return render_template('multi.html')


@app.route('/segment.html', methods=['GET', 'POST'])
def segment():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'Please select an image file.'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Please select an image file.'}), 400

        # Decode uploaded image to BGR numpy array
        file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
        img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)   # ensures BGR uint8

        if img is None:
            return jsonify({'error': 'Failed to decode image.'}), 400

        dt.readImage(img)
        dt.removeNoise()
        dt.displayDisease()

        tumor_percentage = dt.calculateTumorPercentage()

        # Encode result image as base64 JPEG
        success, buf = cv.imencode('.jpg', dt.getImage())
        if not success:
            return jsonify({'error': 'Failed to encode result image.'}), 500

        img_base64 = base64.b64encode(buf.tobytes()).decode('utf-8')
        
        # Connect segmentation with classification
        image_pil = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        processed_multi = preprocess_multiClass_image(image_pil)
        prediction = multiModel.predict(processed_multi)
        predicted_class = np.argmax(prediction)
        
        labels = {0: 'Glioma', 1: 'Meningioma', 2: 'Pituitary'}
        tumor_type = labels.get(predicted_class, 'Unknown')
        
        # Calculate actual percent (100 - background percent)
        actual_percent = 100 - float(tumor_percentage)
        severity = "High" if actual_percent >= 30 else "Low"
        
        return jsonify({
            'img_base64': img_base64, 
            'tumor_percentage': tumor_percentage,
            'tumor_type': tumor_type,
            'severity': severity
        })

    return render_template('segment.html')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
