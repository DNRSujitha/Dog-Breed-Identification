import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pickle

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Use the retrained model
MODEL_PATH = 'models/dog_breed_model_retrained.h5'
CLASS_INDICES_PATH = 'models/class_indices_complete.pkl'

# Global variables
model = None
class_names = []

def load_model_and_classes():
    """Load the retrained model and class names"""
    global model, class_names
    
    print("="*50)
    print("🚀 Starting Dog Breed Classifier")
    print("="*50)
    print(f"TensorFlow version: {tf.__version__}")
    
    # Load class indices
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, 'rb') as f:
            class_indices = pickle.load(f)
        
        # Create class names list
        class_names = [None] * len(class_indices)
        for breed_name, idx in class_indices.items():
            class_names[idx] = breed_name.replace('_', ' ')
        
        print(f"\n✅ Loaded {len(class_names)} dog breeds:")
        for i, name in enumerate(class_names):
            print(f"   {i}. {name}")
    else:
        print("❌ Class indices not found")
        return False
    
    # Load model
    if os.path.exists(MODEL_PATH):
        print(f"\n🔄 Loading model from {MODEL_PATH}...")
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("✅ Model loaded successfully!")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    else:
        print(f"❌ Model not found at {MODEL_PATH}")
        print("\nPlease run retrain_model.py first to create the model.")
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_breed(img_path):
    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(160, 160))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_3_breeds = []
        for idx in top_3_indices:
            if idx < len(class_names) and class_names[idx]:
                top_3_breeds.append({
                    'breed': class_names[idx],
                    'confidence': float(predictions[idx])
                })
        
        result = {
            'breed': class_names[top_3_indices[0]],
            'confidence': float(predictions[top_3_indices[0]]),
            'top_3': top_3_breeds
        }
        
        print(f"\n✅ Prediction: {result['breed']} ({result['confidence']:.2%})")
        return result
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = predict_breed(filepath)
        
        if result:
            return jsonify({
                'success': True,
                'breed': result['breed'],
                'confidence': result['confidence'],
                'top_3': result['top_3'],
                'image_url': f'/static/uploads/{filename}'
            })
        else:
            return jsonify({'error': 'Prediction failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/breeds', methods=['GET'])
def get_breeds():
    if class_names:
        return jsonify({'breeds': class_names})
    return jsonify({'error': 'Breeds not loaded'}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    if load_model_and_classes():
        print("\n" + "="*50)
        print("✅ SERVER READY!")
        print("📍 http://127.0.0.1:5000")
        print("="*50)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load model")