from flask import Flask, render_template, request, redirect, url_for
import cv2 as cv
import numpy as np
import base64
import io
from DisplayDisease import DisplayTumor


app = Flask(__name__)
dt = DisplayTumor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    img = cv.imdecode(np.fromstring(file.read(), np.uint8), cv.IMREAD_UNCHANGED)
    dt.readImage(img)
    dt.removeNoise()
    tumor_percentage = dt.displayTumor() # Get the tumor percentage value from DisplayTumor
    img_str = cv.imencode('.jpg', dt.getImage())[1].tostring()
    img_base64 = base64.b64encode(img_str).decode()
    return render_template('result.html', img_data=img_base64, tumor_percentage=tumor_percentage) # Pass the tumor percentage value to render_template


if __name__ == '__main__':
    app.run(debug=True)
