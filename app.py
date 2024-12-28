from flask import Flask, request, render_template, jsonify, send_file
import os
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

#Load the YOLOv8 model thats trained
model = YOLO('D:/Random_Projects/Tubes Viskom/vehicle_detection/full_data_training_fathan_1object2/weights/best.pt')

#Preprocessing functions to apply to images that have been inputted
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
def gamma_correction(img, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)
def normalize_image(img):
    img = img / 255.0
    return (img * 255).astype(np.uint8)
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}") 
    img = apply_clahe(img)
    img = gamma_correction(img)
    img = normalize_image(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img_3ch = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    return gray_img_3ch


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        #Check the file that is uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        #Save uploaded file to save the results
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)

        #Load the original image
        original_image = cv2.imread(file_path)
        if original_image is None:
            raise ValueError(f"Failed to load image: {file_path}")

        #Preprocess the uploaded image
        preprocessed_image = preprocess_image(file_path)
        preprocessed_path = os.path.join(upload_dir, f'preprocessed_{file.filename}')
        cv2.imwrite(preprocessed_path, preprocessed_image)

        #Perform vehicle detection using the trained YOLOv8 model
        results = model(preprocessed_path)

        #Extract bounding box and class info from the object detection
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  #Bounding box coordinates
            confidence = box.conf[0]  #Confidence score
            class_id = int(box.cls[0])  #Class ID

            #Draw the bounding box and label on the original image
            label = f"{model.names[class_id]} {confidence:.2f}"
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        #Save the result on the original image
        result_path = os.path.join(upload_dir, f'result_{file.filename}')
        cv2.imwrite(result_path, original_image)

        return send_file(result_path, mimetype='image/png')

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
