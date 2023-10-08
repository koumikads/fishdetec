from flask import Flask, jsonify, request
import cv2
import time
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load YOLOv5 model
model = tf.saved_model.load('best.pt/')


# Define endpoint to detect fish
@app.route('/detect-fish', methods=['POST'])
def detect_fish():

    # Capture image from webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    # Convert image to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess image
    # Preprocess image
    img = tf.image.resize(img, [640, 640])
    img = img / 255.0
    img = tf.expand_dims(img, 0)

    # Run YOLOv5 model
    outputs = model(img)
    results = outputs['tf_op_layer_concat_10'].numpy()[0]

    # Filter for fish class
    fish_results = [r for r in results if r[4] == 0]

    # Return number of fish detected
    return jsonify({'num_fish': len(fish_results)})


if __name__ == '__main__':
    app.run(debug=True)
