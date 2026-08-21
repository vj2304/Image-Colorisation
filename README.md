# 🎨 Image Colorisation

A deep learning–based computer vision application that automatically converts **grayscale images into colorized images**. The project provides an interactive web interface where users can upload a black-and-white image and obtain its colorized version.

## 🚀 Features

* 🖼️ Convert grayscale images into colorized images
* 🤖 Deep learning–based color prediction
* ⚙️ Automated image preprocessing and model inference
* 🌐 Flask-based web interface
* 📤 Upload images directly through the browser
* 📥 View the generated colorized output
* 🔄 End-to-end pipeline from image upload to colorized result

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Machine Learning:** Deep Learning / Neural Networks
* **Computer Vision:** Image Processing
* **Backend:** Flask
* **Frontend:** HTML, CSS
* **Model:** Pre-trained image colorization model

## 📂 Project Structure

```text
Image-Colorisation/
│
├── app.py                 # Flask application
├── model.py               # Model architecture and inference logic
├── model/                 # Trained model files
│── index.html             # Image upload page
│── result.html            # Colorized image output page
├── static/                # Static assets
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 🔄 How It Works

The application follows the following pipeline:

```text
Input Grayscale Image
        ↓
Image Upload
        ↓
Image Preprocessing
        ↓
Deep Learning Model
        ↓
Color Prediction
        ↓
Post-processing
        ↓
Colorized Image
        ↓
Web Interface
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/vj2304/Image-Colorisation.git
cd Image-Colorisation
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Running the Application

Start the Flask application:

```bash
python app.py
```

Open your browser and navigate to:

```text
http://127.0.0.1:5000
```

Upload a grayscale image and the application will generate the corresponding colorized output.

## 📸 Example

### Input

A grayscale/black-and-white image is uploaded through the web interface.

### Output

The deep learning model predicts suitable colors and generates a colorized version of the input image.

> **Note:** Model output quality depends on the input image and the training/model architecture. Colorization is an inherently predictive task, so generated colors may not always represent the original colors.

## 🧠 Key Concepts

This project demonstrates practical applications of:

* Deep Learning
* Computer Vision
* Image Processing
* Image-to-Image Transformation
* Neural Network Inference
* Model Deployment
* Flask Web Development

## 🔮 Future Improvements

* Improve colorization quality using more advanced architectures
* Train on larger and more diverse datasets
* Add GPU acceleration for faster inference
* Add batch image processing
* Provide drag-and-drop image upload
* Add before/after image comparison
* Deploy the application using a cloud platform
* Add quantitative image-quality evaluation metrics

## 👨‍💻 Author

**VJ2304**

GitHub:
https://github.com/vj2304
