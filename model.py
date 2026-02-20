import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import time

# OPTIMIZED CONFIGURATION FOR 1000 IMAGES
IMG_SIZE = 160  # Smaller size = faster training
BATCH_SIZE = 32 
EPOCHS = 20  
DATASET_PATH = '/content/drive/MyDrive/dog-breed-dataset'  # Update this to your path
MODEL_PATH = '/content/drive/MyDrive/dog_breed_model.h5'
CLASS_INDICES_PATH = '/content/drive/MyDrive/class_indices.pkl'

# Enable mixed precision for faster training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def prepare_data():
    """Prepare data with strong augmentation for small dataset"""
    
    # Check dataset path
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Path {DATASET_PATH} not found!")
        print("Please update DATASET_PATH to correct location")
        return None, None, None
    
    # STRONG DATA AUGMENTATION (crucial for small dataset)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,  # More rotation
        width_shift_range=0.3,  # More shift
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],  # Add brightness variation
        fill_mode='nearest',
        validation_split=0.2  # 80% train, 20% validation
    )
    
    # Validation data - only rescaling
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    print("Loading dataset...")
    
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
    
    # Save class indices
    class_indices = train_generator.class_indices
    with open(CLASS_INDICES_PATH, 'wb') as f:
        pickle.dump(class_indices, f)
    
    print(f"\n✅ Training samples: {train_generator.samples}")
    print(f"✅ Validation samples: {validation_generator.samples}")
    print(f"✅ Number of breeds: {len(class_indices)}")
    print(f"✅ Breeds: {list(class_indices.keys())}")
    
    return train_generator, validation_generator, class_indices

def create_model(num_classes):
    """Create model with strong regularization for small dataset"""
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model with dropout for regularization
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)  # Increased dropout
    x = layers.Dense(256, activation='relu', kernel_regularizer='l2')(x)  # L2 regularization
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer='l2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base_model

def train_model():
    """Train with early stopping to prevent overfitting"""
    
    # Prepare data
    train_generator, validation_generator, class_indices = prepare_data()
    if train_generator is None:
        return None, None
    
    num_classes = len(class_indices)
    
    # Create model
    model, base_model = create_model(num_classes)
    
    # Print model summary
    model.summary()
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks for small dataset - REMOVED 'workers' parameter
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,  # Stop if no improvement for 5 epochs
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train - REMOVED workers and use_multiprocessing parameters
    print("\n🚀 Starting training...")
    start_time = time.time()
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\n✅ Initial training completed in {training_time//60:.0f}m {training_time%60:.0f}s")
    
    # Fine-tuning with very low learning rate
    print("\n🔄 Starting fine-tuning...")
    base_model.trainable = True
    
    # Freeze early layers
    for layer in base_model.layers[:50]:
        layer.trainable = False
    
    # Recompile with very low learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00001),  # 10x smaller LR
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training - REMOVED workers parameter
    history_fine = model.fit(
        train_generator,
        epochs=EPOCHS + 5,
        initial_epoch=history.epoch[-1] + 1,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")
    
    # Print final accuracy
    val_acc = max(history.history['val_accuracy'] + history_fine.history['val_accuracy'])
    print(f"\n🎯 Best validation accuracy: {val_acc:.2%}")
    
    return model, history

if __name__ == "__main__":
    # Mount Google Drive first (if using Colab)
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Train the model
    model, history = train_model()
    
    print("\n✅✅✅ TRAINING COMPLETE! ✅✅✅")