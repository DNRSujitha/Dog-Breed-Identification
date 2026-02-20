# rebuild_model.py
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
import pickle
import numpy as np
import os

print("="*60)
print("REBUILDING MODEL FOR TF 2.15")
print("="*60)

# Load class indices
print("\n📁 Loading class indices...")
with open('models/class_indices_complete.pkl', 'rb') as f:
    class_indices = pickle.load(f)

num_classes = len(class_indices)
print(f"✅ Loaded {num_classes} breeds")

# Create model architecture in TF 2.15 compatible format
print("\n🏗️ Building model architecture...")

# Load MobileNetV2 base
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(160, 160, 3)
)
base_model.trainable = False

# Build the model WITHOUT using batch_shape parameter
inputs = tf.keras.Input(shape=(160, 160, 3))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)

print("✅ Model architecture built")
print(f"   Input shape: {model.input_shape}")
print(f"   Output shape: {model.output_shape}")

# Try to load weights from various files
print("\n🔍 Attempting to load weights...")

weight_files = [
    'models/dog_breed_weights.weights.h5',
    'models/dog_breed_model_final.weights.h5',
    'models/dog_breed_model_final.h5',
    'models/dog_breed_model_tf215.h5'
]

weights_loaded = False
for wf in weight_files:
    if os.path.exists(wf):
        print(f"   Trying: {wf}")
        try:
            # Try loading as weights
            model.load_weights(wf)
            print(f"   ✅ Weights loaded successfully from {wf}")
            weights_loaded = True
            break
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            
            # Try loading as full model and extract weights
            try:
                print(f"      Attempting to extract weights from full model...")
                temp_model = tf.keras.models.load_model(wf, compile=False)
                model.set_weights(temp_model.get_weights())
                print(f"      ✅ Weights extracted successfully")
                weights_loaded = True
                break
            except:
                pass

if not weights_loaded:
    print("\n❌ Could not load weights from any file")
    print("\nCreating a new untrained model (will predict randomly)")
else:
    print("\n✅ Model ready with loaded weights")

# Save the rebuilt model in TF 2.15 compatible format
print("\n💾 Saving rebuilt model...")
model.save('models/dog_breed_model_rebuilt.h5')
print("✅ Model saved as: models/dog_breed_model_rebuilt.h5")

# Test the model
print("\n🧪 Testing model with random input...")
test_input = tf.random.normal((1, 160, 160, 3))
output = model(test_input)
print(f"✅ Model inference works")
print(f"   Output shape: {output.shape}")
print(f"   Prediction probabilities sum: {tf.reduce_sum(output[0]).numpy():.4f}")

# Show top prediction for random input
pred_probs = output[0].numpy()
top_idx = np.argmax(pred_probs)
breed_names = list(class_indices.keys())
print(f"\n📊 Random test prediction: {breed_names[top_idx]} ({pred_probs[top_idx]*100:.2f}%)")

print("\n" + "="*60)
print("REBUILD COMPLETE")
print("="*60)