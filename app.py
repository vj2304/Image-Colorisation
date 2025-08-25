import os
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import cv2

# Define the ColorizationNet model class (unchanged)
class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv4 = nn.Conv2d(128, 3, kernel_size=5, stride=1, padding=4, dilation=2)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x

# Initialize Flask app and directories
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "colorization_net_model.pkl"
model = torch.load(model_path, map_location=device)
model.eval()

# Preprocessing function
def preprocess_image(image_path):
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_size = original_image.shape[:2]
    image = original_image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    return torch.tensor(image, dtype=torch.float32).to(device), original_size

# Postprocessing function
def postprocess_output(output_tensor, original_size):
    output = output_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    output = (output * 255).clip(0, 255).astype('uint8')
    output_resized = cv2.resize(output, (original_size[1], original_size[0]))
    return output_resized

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        input_tensor, original_size = preprocess_image(filepath)
        with torch.no_grad():
            output_tensor = model(input_tensor)

        colorized_image = postprocess_output(output_tensor, original_size)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"colorized_{filename}")
        cv2.imwrite(output_path, cv2.cvtColor(colorized_image, cv2.COLOR_RGB2BGR))

        return render_template(
            'result.html',
            grayscale_image=filename,
            colorized_image=f"colorized_{filename}"
        )
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Optional route for serving additional static files
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
