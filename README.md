AI Based Multi-class Ensemble System For Cervical Cancer Detection

This project is an AI-Based Multi-Class Ensemble System designed to detect cervical cancer cell types from microscope images.
The system uses a deep learning ensemble model with 95% accuracy, combining EfficientNetB0 and Fine-Tuned MobileNetV2, and is deployed as a web application built with Flask.

📌 Overview

The system classifies cervical cell images into:

Malignant# TEAM.NO.49:AI Based Multi-Class Ensemble Learning System for Cervical Cancer Detection

## About

This project focuses on detecting cervical cancer using a deep learning ensemble architecture. The system classifies cervical cytology images into:

1.Normal

2.Precancerous

3.Malignant


The solution integrates EfficientNetB0 and MobileNetV2 (fine-tuned), combined using soft probability voting, achieving a final accuracy of 95%.
A Flask-based web app enables users to upload images and receive predictions, along with a downloadable medical-style PDF report.

## Features

📤 Upload cervical cell image

🤖 Real-time prediction using ensemble deep learning

📊 Confidence score for each class

📝 Enter patient details before prediction

📄 Downloadable PDF medical report

## Development Requirements

<img width="692" height="317" alt="1" src="https://github.com/user-attachments/assets/ee24ef97-3d2a-4524-9c1b-45c3d2f4f603" />

## System Architecture

![2](https://github.com/user-attachments/assets/9e383925-7662-4de2-a561-c26713d4da8c)

📌 Ensemble Formula:
final_probabilities = (EfficientNetB0_output + MobileNetV2_output) / 2

## Methodology
### 1. Data Preprocessing

i) The images from the Mendeley Cervical Cytology Dataset were cleaned by removing corrupted or unreadable files.

ii) All images were resized to 224 × 224 px, normalized, and converted into a consistent RGB format suitable for CNN processing.

iii) Data augmentation techniques such as rotation, zoom, brightness shift, and horizontal flip were applied to improve generalization and reduce overfitting.

### 2. Model Training

i) Two deep learning models were used for feature extraction and classification:

1.EfficientNetB0 (Pretrained on ImageNet)

2.Fine-Tuned MobileNetV2

ii) The outputs of both models were combined using a soft probability voting ensemble technique, forming a final deployable model named:
ensemble_model.keras

iii) The model was trained in Google Colab using GPU acceleration with Adam optimizer, categorical cross-entropy loss, and early stopping to prevent overfitting.

### 3. Model Evaluation

Evaluation metrics included: accuracy, precision, recall, F1-score, and confusion matrix.

The ensemble model demonstrated improved performance across all classes compared to individual models.

The final deployed model achieved:

<img width="507" height="346" alt="3" src="https://github.com/user-attachments/assets/234f3fdb-1eb5-4ddc-8efc-bc6483291d30" />

<img width="676" height="215" alt="4" src="https://github.com/user-attachments/assets/a06fd009-e392-47c7-8b8c-14e7731f9441" />

### 4. Setup Instructions
#### Run the Flask Web App:
```
.\venv\Scripts\Activate
python app.py
```
#### Access Web Interface:
```
http://127.0.0.1:8000
http://172.20.10.5:8000
```
## Key Model Implementation Code
```
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report

# IMAGE SIZE & CLASS LABELS
img_size = (224, 224)
class_labels = ["Normal", "Precancerous", "Malignant"]

# MODEL 1: EfficientNetB0
base1 = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x1 = GlobalAveragePooling2D()(base1.output)
x1 = Dropout(0.4)(x1)
out1 = Dense(3, activation="softmax")(x1)
eff_model = Model(inputs=base1.input, outputs=out1)

# MODEL 2: Fine-Tuned MobileNetV2
base2 = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x2 = GlobalAveragePooling2D()(base2.output)
x2 = Dropout(0.4)(x2)
out2 = Dense(3, activation="softmax")(x2)
mobile_model = Model(inputs=base2.input, outputs=out2)

# Load trained models
eff_model = tf.keras.models.load_model("efficientnet_model.h5")
mobile_model = tf.keras.models.load_model("mobilenet_finetuned.h5")

# ENSEMBLE LAYER (Soft Voting)
input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
eff_output = eff_model(input_layer)
mob_output = mobile_model(input_layer)

ensemble_output = tf.keras.layers.Average()([eff_output, mob_output])
ensemble_model = Model(inputs=input_layer, outputs=ensemble_output)

# Save Final Model
ensemble_model.save("ensemble_model.keras")
print("Final Ensemble Model Saved Successfully")

# IMAGE PREPROCESSING & PREDICTION
def preprocess_image(path):
    img = image.load_img(path, target_size=img_size)
    img = image.img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

def predict_image(img_path):
    img = preprocess_image(img_path)
    eff_pred = eff_model.predict(img)
    mob_pred = mobile_model.predict(img)
    final_prob = (eff_pred + mob_pred) / 2
    pred_class = np.argmax(final_prob)
    print("\nPredicted Class:", class_labels[pred_class])
    print("Confidence Score:", round(float(np.max(final_prob)) * 100, 2), "%")
```

## Results
The final ensemble model achieved an accuracy of 95%, providing strong performance in classifying cervical cytology images across all three classes (Normal, Precancerous, Malignant).

This system enables early-stage detection, which may support medical professionals and improve preventive healthcare outcomes.

## Output

#### Web-page asking for input from user

<img width="1920" height="1080" alt="5" src="https://github.com/user-attachments/assets/e21ff2d8-89cd-4469-b462-9ce465741c55" />

<img width="1920" height="1080" alt="6" src="https://github.com/user-attachments/assets/e7dca178-195a-4242-9e54-f4b260368c87" />

#### Web-page displays the result

<img width="1920" height="1080" alt="7" src="https://github.com/user-attachments/assets/6daef34a-edc8-4aa4-abd0-e2e387a7243a" />

## Future Enhancements

🔹 Store patient history using Firebase/MongoDB

🔹 Add batch image prediction

🔹 Deploy inference on GPU cloud

## References

[1] M. Tan and Q. V. Le, “EfficientNet: Rethinking model scaling for convolutional neural 
networks,” Proceedings of the 36th International Conference on Machine Learning (ICML), 2019.

[2] A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and 
H. Adam, “MobileNets: Efficient convolutional neural networks for mobile vision applications,” 
arXiv preprint arXiv:1704.04861, 2017. 

[3] J. Zhang, F. Xie, Y. Qian, and X. Xie, “Cervical cancer diagnosis using deep convolutional 
neural networks,” IEEE Access, vol. 8, pp. 91245–91256, 2020. 

[4] K. Zhou and X. Chen, “Ensemble learning for medical image classification: A comprehensive 
review,” Medical Image Analysis, 2021.

Precancerous

Normal

The user provides basic details, uploads an image, and receives the prediction along with a downloadable PDF medical report.

The web app features a clean pink-themed UI, and is fully deployable using Render or Railway for a permanent URL.

⭐ Features

✔ Deep Learning Ensemble Model (EfficientNet + MobileNetV2)

✔ 95% accuracy on Mendeley Cervical Cancer 3-Class Dataset

✔ User-friendly web form (Name, Age, Gender, Phone, City, etc.)

✔ Image Upload & Real-Time Prediction

✔ Confidence Scores for all 3 classes

✔ PDF Report Generation (ReportLab)

✔ Deployment ready with Dockerfile and requirements.txt

✔ Runs locally in VS Code

📁 Project Structure
cervical-web-app/
│
├─ app.py                     # Flask backend
├─ requirements.txt           # All dependencies
├─ Dockerfile                 # Deployment container
├─ README.md                  
├─ models/
│   └─ ensemble_model.keras   # Final deployable model
│
├─ templates/
│   ├─ index.html             # Input form page
│   └─ result.html            # Result + PDF download page
│
└─ static/
    ├─ css/
    │   └─ styles.css         # Pink UI theme
    └─ uploads/               # Uploaded images + generated PDFs

🧠 Ensemble Model Details
1. EfficientNetB0

Pretrained on ImageNet

Good at identifying Normal cells

2. MobileNetV2 (Fine-Tuned)

Fine-tuned top layers

Best at detecting Malignant + Precancerous

3. Soft Voting Logic
final_prob = (efficientnet_prob + mobilenet_prob) / 2


The combined ensemble is saved as a single .keras model for easy deployment.

🧰 Technologies Used
Backend

Python

Flask

TensorFlow

NumPy

Pillow

ReportLab (PDF generation)

Frontend

HTML

CSS

Bootstrap

Deployment

Docker

Gunicorn

Render / Railway

🚀 How to Run This Project (VS Code)
1️⃣ Clone the project
git clone https://github.com/yourusername/cervical-web-app.git
cd cervical-web-app

2️⃣ Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate

Linux / macOS
python3 -m venv venv
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Web App
python app.py


Open the browser at:

http://127.0.0.1:8501/


Upload an image → Get prediction → Download PDF.

🌍 Deployment (Render / Railway)

Push project to GitHub

Create New Web Service

Connect GitHub repo

Render auto-detects Dockerfile

Deploy → Get permanent public URL

Example:

https://cervical-cancer-detection.onrender.com

📄 PDF Report Includes:

Patient Details

Uploaded Image

Final Prediction

Confidence Scores

Date & Timestamp

Model used: EfficientNetB0 + MobileNetV2 Ensemble

🔮 Future Improvements

Add Grad-CAM heatmaps

Add patient history database

Add email PDF feature

Deploy inference on GPU

Add doctors login dashboard

✨ Acknowledgement

Special thanks to the Mendeley LBC dataset and open-source deep learning community.
