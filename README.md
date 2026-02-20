# Dog-Breed-Identification
# 🐕 Dog Breed Classifier

An AI-powered web application that identifies dog breeds from uploaded photos using transfer learning.

## Features

- Upload a dog photo and get instant breed identification
- Top 3 predictions with confidence scores
- Responsive design for mobile and desktop
- Beautiful UI with smooth animations

## Tech Stack

- **Backend**: Flask, TensorFlow 2.15
- **Frontend**: HTML, CSS, JavaScript
- **Model**: MobileNetV2 with transfer learning
- **Dataset**: 10 dog breeds (Beagle, Boxer, Bulldog, etc.)

## Project Structure
``` sh
dog-breed-identification/
├── app.py # Main Flask application
├── model.py # Original model training script
├── rebuild_model.py # TF 2.15 compatibility rebuild
├── retrain_model.py # Quick retraining script
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore file
├── README.md # This file
├── templates/ # HTML templates
│ └── index.html
├── static/ # CSS, JS, uploads
│ ├── css/
│ │ └── style.css
│ ├── js/
│ │ └── script.js
│ └── uploads/ # User uploaded images
└── models/ # Model files (download separately)
└── class_indices_complete.pkl # Class names
```
## Installation

**1. Clone the repository:**

```sh
git clone https://github.com/DNRSujitha/Dog-Breed-Identification.git
```
cd dog-breed-classifier'''
**2.Install dependencies:**
```sh
pip install -r requirements.txt
```

Download the trained model from the Releases page

Place dog_breed_model_retrained.h5 in the models/ folder

**3.Run the app:**
```sh
python app.py'
```
Open http://localhost:5000 in your browser

## Training Your Own Model
**Option 1: Quick Retraining**
```sh

python retrain_model.py'
```
**Option 2: Full Training**
```sh
python model.py
```

**Option 3: Rebuild for Compatibility**
```sh
python rebuild_model.py
```

## Output:
![My App Screenshot](https://github.com/user-attachments/assets/c795d078-6058-4d8e-9535-ff9a5b1eb309)


## Dataset
The model was trained on 10 dog breeds:

1. Beagle
2. Boxer
3. Bulldog
4. Dachshund
5. German Shepherd
6. Golden Retriever
7. Labrador Retriever
8. Poodle
9. Rottweiler
10. Yorkshire Terrier

## Licence
MIT Licence
