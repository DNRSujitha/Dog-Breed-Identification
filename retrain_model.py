# retrain_model.py
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

print("="*60)
print("RETRAINING MODEL IN TF 2.15")
print("="*60)
print(f"TensorFlow version: {tf.__version__}")

# Configuration
IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 10  # Quick retraining

# Update this path to where your dataset is located
DATASET_PATH = 'C:/Users/sailu/dog-breed-classifier/dataset'  # Change this to your dataset path
CLASS_INDICES_PATH = 'models/class_indices_complete.pkl'
MODEL_SAVE_PATH = 'models/dog_breed_model_retrained.h5'

# Check if dataset exists
if not os.path.exists(DATASET_PATH):
    print(f"\n❌ Dataset not found at {DATASET_PATH}")
    print("Please update DATASET_PATH to point to your dataset folder")
    exit(1)

# Load class indices to verify
with open(CLASS_INDICES_PATH, 'rb') as f:
    class_indices = pickle.load(f)
print(f"\n✅ Loaded {len(class_indices)} breed classes")

# Data preparation with augmentation
print("\n🔄 Preparing data...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training generator
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Validation generator
validation_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"\n📊 Training samples: {train_generator.samples}")
print(f"📊 Validation samples: {validation_generator.samples}")

# Build model
print("\n🏗️ Building model...")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_indices), activation='softmax')(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model built")

# Train
print("\n🚀 Starting training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    verbose=1
)

# Save model
print("\n💾 Saving model...")
model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")

# Show final accuracy
final_acc = history.history['val_accuracy'][-1]
print(f"\n📊 Final validation accuracy: {final_acc:.2%}")

print("\n" + "="*60)
print("RETRAINING COMPLETE")
print("="*60)