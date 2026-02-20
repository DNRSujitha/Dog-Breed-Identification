// Configuration
const API_URL = window.location.origin;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    loadBreedsList();
});

function setupEventListeners() {
    // File input change event
    document.getElementById('imageInput').addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            previewImage(e.target.files[0]);
        } else {
            hidePreview();
        }
    });
    
    // Smooth scrolling for navigation links
    document.querySelectorAll('.nav-links a').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            if (targetSection) {
                targetSection.scrollIntoView({ behavior: 'smooth' });
            }
            
            // Update active class
            document.querySelectorAll('.nav-links a').forEach(l => l.classList.remove('active'));
            this.classList.add('active');
        });
    });
}

function previewImage(file) {
    const reader = new FileReader();
    const previewContainer = document.getElementById('imagePreviewContainer');
    const preview = document.getElementById('imagePreview');
    
    reader.onload = function(e) {
        preview.src = e.target.result;
        previewContainer.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function hidePreview() {
    document.getElementById('imagePreviewContainer').style.display = 'none';
    document.getElementById('imagePreview').src = '';
}

function predictBreed() {
    const fileInput = document.getElementById('imageInput');
    const resultDiv = document.getElementById('result');
    const loadingDiv = document.getElementById('loading');
    const predictionDetails = document.getElementById('predictionDetails');
    
    // Clear previous results
    resultDiv.innerHTML = '';
    predictionDetails.style.display = 'none';
    
    if (fileInput.files.length === 0) {
        showError('Please upload an image first.');
        return;
    }
    
    const file = fileInput.files[0];
    
    // Check file type
    if (!file.type.startsWith('image/')) {
        showError('Please upload a valid image file.');
        return;
    }
    
    // Check file size (max 16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size too large. Maximum size is 16MB.');
        return;
    }
    
    // Show loading indicator
    loadingDiv.style.display = 'block';
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    // Send to backend
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'Prediction failed');
            });
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Prediction failed');
        }
    })
    .catch(error => {
        showError(error.message || 'Network error. Please try again.');
        console.error('Error:', error);
    })
    .finally(() => {
        loadingDiv.style.display = 'none';
    });
}

function displayResults(data) {
    const resultDiv = document.getElementById('result');
    const predictionDetails = document.getElementById('predictionDetails');
    
    // Format confidence percentage
    const confidencePercent = (data.confidence * 100).toFixed(2);
    
    // Display main result with confidence color
    let confidenceColor = '#4a6cf7';
    if (data.confidence > 0.8) confidenceColor = '#28a745';
    else if (data.confidence > 0.6) confidenceColor = '#ffc107';
    else if (data.confidence < 0.4) confidenceColor = '#dc3545';
    
    resultDiv.innerHTML = `
        <div class="success-result">
            <h3>🐕 Predicted Breed:</h3>
            <h2 style="color: ${confidenceColor};">${data.breed}</h2>
            <div class="confidence-circle">
                <svg viewBox="0 0 36 36" class="circular-chart">
                    <path class="circle-bg"
                        d="M18 2.0845
                        a 15.9155 15.9155 0 0 1 0 31.831
                        a 15.9155 15.9155 0 0 1 0 -31.831"
                        fill="none"
                        stroke="#eee"
                        stroke-width="3"/>
                    <path class="circle"
                        stroke-dasharray="${data.confidence * 100}, 100"
                        d="M18 2.0845
                        a 15.9155 15.9155 0 0 1 0 31.831
                        a 15.9155 15.9155 0 0 1 0 -31.831"
                        fill="none"
                        stroke="${confidenceColor}"
                        stroke-width="3"
                        stroke-linecap="round"/>
                    <text x="18" y="20.35" class="percentage">${confidencePercent}%</text>
                </svg>
            </div>
        </div>
    `;
    
    // Display top 3 predictions
    if (data.top_3 && data.top_3.length > 0) {
        const topPredictionsDiv = document.getElementById('topPredictions');
        let topHtml = '<div class="top-3-list">';
        
        data.top_3.forEach((item, index) => {
            const confidence = (item.confidence * 100).toFixed(2);
            const barWidth = item.confidence * 100;
            const medal = index === 0 ? '🥇' : index === 1 ? '🥈' : '🥉';
            
            topHtml += `
                <div class="prediction-item">
                    <div class="prediction-rank">${medal}</div>
                    <div class="prediction-details">
                        <div class="breed-name">${item.breed}</div>
                        <div class="confidence-container">
                            <div class="confidence-bar" style="width: ${barWidth}%"></div>
                            <span class="confidence-text">${confidence}%</span>
                        </div>
                    </div>
                </div>
            `;
        });
        
        topHtml += '</div>';
        topPredictionsDiv.innerHTML = topHtml;
    }
    
    predictionDetails.style.display = 'block';
}

function showError(message) {
    const resultDiv = document.getElementById('result');
    const predictionDetails = document.getElementById('predictionDetails');
    
    resultDiv.innerHTML = `
        <div class="error-message">
            ❌ ${message}
        </div>
    `;
    predictionDetails.style.display = 'none';
}

function loadBreedsList() {
    // Optional: Load breeds list for reference
    fetch('/breeds')
        .then(response => response.json())
        .then(data => {
            if (data.breeds && data.breeds.length > 0) {
                console.log('Available breeds:', data.breeds);
            }
        })
        .catch(error => {
            console.error('Error loading breeds:', error);
        });
}

// Add keyboard support
document.getElementById('imageInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        predictBreed();
    }
});