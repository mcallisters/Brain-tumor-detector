# Brain Tumor Detection System

An end-to-end **AI-powered medical imaging pipeline** that detects brain tumors from MRI scans using deep learning. The system includes data preprocessing, transfer learning with ResNet18, model interpretability with GradCAM, and an interactive Streamlit web application.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🎯 Project Overview

This project implements a complete machine learning pipeline for brain tumor detection:

1. **Data Preprocessing**: Automated image cleaning, duplicate detection, and quality control
2. **Model Training**: Transfer learning with ResNet18 on 1,356 labeled MRI scans
3. **Model Evaluation**: Comprehensive metrics including F1-score, ROC-AUC, confusion matrices
4. **Model Interpretability**: GradCAM visualizations to understand model decisions
5. **Web Deployment**: Interactive Streamlit application for real-time predictions

**⚠️ Disclaimer**: This tool is for educational and research purposes only. It is NOT intended for clinical diagnosis or medical decision-making.

---

## 📊 Dataset

- **Total Images**: 1,356 brain MRI scans
- **Classes**: 
  - `yes` - Brain tumor present
  - `no` - No brain tumor
- **Image Size**: 224×224 pixels (standardized)
- **Split Ratio**: 70% train / 20% validation / 10% test
- **Preprocessing**: Duplicate removal, outlier detection, brightness normalization

---

## 🏗️ Project Structure

```
Brain-Tumor-Detector/
├── data/
│   ├── pg_dataset/                    # Raw dataset (original images)
│   └── brain_tumor_data_preprocessed_all/  # Cleaned & preprocessed images
│       ├── yes/                       # Tumor images
│       └── no/                        # Non-tumor images
├── models/
│   ├── best_model_m1_notebook.pt      # Trained model weights
│   └── test_results.json              # Model performance metrics
├── scripts/
│   ├── preprocessing.py               # Data cleaning & quality control
│   ├── train_model.py                 # Model training pipeline
│   ├── histogram_visualization.py     # Performance analysis & visualization
│   └── gradcam.py                     # Model interpretability (GradCAM)
├── gradcam_visualizations/            # GradCAM output images
├── streamlit_app.py                   # Streamlit web application
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

---

## 🚀 Features

### Data Preprocessing (`preprocessing.ipynb`)
- **Duplicate Detection**: Perceptual hashing to identify and remove near-duplicate images
- **Outlier Removal**: Z-score analysis for brightness anomalies
- **Quality Control**: Low-variance detection to filter blank/corrupted images
- **Standardization**: Resize all images to 224×224 pixels
- **Aspect Ratio Correction**: Center-crop images with extreme aspect ratios
- **Statistical Reporting**: Before/after preprocessing statistics

### Model Training (`brain_tumor_training_resnet18.py`)
- **Architecture**: ResNet18 (pre-trained on ImageNet)
- **Transfer Learning**: Fine-tuned all layers on brain MRI dataset
- **Data Augmentation**: Random flips, rotations, color jitter
- **Class Balancing**: Weighted loss function to handle class imbalance
- **Early Stopping**: Patience-based stopping to prevent overfitting
- **Threshold Tuning**: Optimal decision threshold selection on validation set

### Model Evaluation (`histogram_visualization_prob_resnet18.py`)
- **Confusion Matrix**: True positives, false positives, true negatives, false negatives
- **ROC Curve**: ROC-AUC score visualization
- **Probability Distributions**: Histogram analysis by class
- **Sample Visualization**: Display predictions with confidence scores

### Model Interpretability (`gradcam_brain_tumor_resnet18.py`)
- **GradCAM Heatmaps**: Visualize which regions the model focuses on
- **Batch Processing**: Generate explanations for multiple images
- **Overlay Visualization**: Heatmap overlays on original images
- **Decision Validation**: Verify model is looking at relevant brain regions

### Web Application (`streamlit_app.py`)
- **Single Image Upload**: Upload and analyze individual MRI scans
- **Batch Processing**: Analyze multiple images simultaneously
- **Probability Visualization**: Interactive probability charts
- **Model Metrics**: Display F1-score, ROC-AUC, sensitivity, specificity
- **User-Friendly Interface**: No coding required

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Brain-Tumor-Detector.git
cd Brain-Tumor-Detector
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages
```txt
torch>=2.0.0
torchvision>=0.15.0
streamlit>=1.28.0
pillow>=9.0.0
numpy>=1.24.0
opencv-python>=4.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
pandas>=2.0.0
imagehash>=4.3.0
networkx>=3.1.0
```

---

## 🎓 Usage Guide

### Step 1: Data Preprocessing

Preprocess raw MRI images (remove duplicates, outliers, standardize size):

```bash
python scripts/preprocessing.ipynb
```

**What it does**:
- Scans `data/pg_dataset/` for raw images
- Applies duplicate detection (perceptual hashing)
- Removes outliers (Z-score brightness analysis)
- Filters low-variance (blank) images
- Resizes to 224×224 pixels
- Saves cleaned data to `data/brain_tumor_data_preprocessed_all/`

**Configuration** (edit in `preprocessing.py`):
```python
SIMILARITY_THRESHOLD = 2      # Perceptual hash distance (lower = stricter)
Z_SCORE_THRESHOLD = 5.0       # Brightness outlier threshold
LOW_VARIANCE_THRESHOLD = 10   # Minimum pixel variance
TARGET_SIZE = (224, 224)      # Output image size
```

### Step 2: Train the Model

Train ResNet18 on preprocessed data:

```bash
python scripts/train_model.py
```

**What it does**:
- Loads preprocessed images from `data/brain_tumor_data_preprocessed_all/`
- Splits data: 70% train, 20% validation, 10% test
- Applies data augmentation (flips, rotations, color jitter)
- Trains ResNet18 with weighted loss for class imbalance
- Saves best model to `models/best_model_m1_notebook.pt`
- Performs threshold tuning on validation set
- Evaluates on test set and saves metrics to `test_results.json`

**Hyperparameters** (edit in `train_model.py`):
```python
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 12
LEARNING_RATE = 1e-4
PATIENCE = 5              # Early stopping patience
```

**Training Output**:
```
Epoch [1/12] Train Loss: 0.4523, Train Acc: 0.7891, Val F1 (yes): 0.8234
✓ Model saved! New best F1: 0.8234
...
Optimal threshold: 0.50, F1-score: 0.8456
Test Set F1 (yes class): 0.8312
```

### Step 3: Visualize Model Performance

Generate performance visualizations and analyze predictions:

```bash
python scripts/histogram-visualization-prob-resnet18.py
```

**What it does**:
- Loads trained model and preprocessed data
- Generates ROC curve with AUC score
- Creates probability distribution histograms
- Visualizes true positives, true negatives, false positives, false negatives
- Displays sample predictions with confidence scores

**Output**: Interactive matplotlib visualizations showing model performance

### Step 4: Generate GradCAM Explanations

Visualize what the model "sees" when making predictions:

```bash
python scripts/gradcam.py
```

**What it does**:
- Loads trained model
- Generates GradCAM heatmaps for sample images
- Creates overlay visualizations (original + heatmap)
- Saves visualizations to `gradcam_visualizations/`
- Shows which brain regions influence predictions

**Configuration** (edit in `gradcam.py`):
```python
NUM_SAMPLES = 5           # Images to visualize per class
OUTPUT_DIR = 'gradcam_visualizations/'
```

**Output**:
```
gradcam_visualizations/
├── yes_1_image123_gradcam.png
├── yes_2_image456_gradcam.png
├── no_1_image789_gradcam.png
└── ...
```

### Step 5: Run Web Application

Launch the interactive Streamlit app:

```bash
streamlit run streamlit_app.py
```

**What it does**:
- Opens web interface at `http://localhost:8501`
- Allows single or batch image upload
- Displays predictions with confidence scores
- Shows probability visualizations
- Provides model performance metrics

**Usage**:
1. Upload MRI image(s) (PNG, JPG, JPEG)
2. View prediction: "Tumor Detected" or "No Tumor"
3. See confidence score and probability chart
4. Download results (optional)

---

## 🧠 Model Architecture

### Base Model: ResNet18
- **Pre-trained Weights**: ImageNet (1000 classes)
- **Transfer Learning Approach**: Replace final fully connected layer
- **Custom Head**: `Linear(512 → 2)` for binary classification

### Architecture Diagram
```
Input (224×224×3 RGB image)
    ↓
[ResNet18 Feature Extractor]
├── Conv Layer 1 (64 filters)
├── Residual Block 1 (64 filters)
├── Residual Block 2 (128 filters)
├── Residual Block 3 (256 filters)
└── Residual Block 4 (512 filters)
    ↓
[Global Average Pooling]
    ↓
[Fully Connected Layer] (512 → 2)
    ↓
[Softmax]
    ↓
Output: [P(tumor), P(no tumor)]
```

### Why Transfer Learning?
- **Pre-trained Features**: ResNet18 learned general visual features from 1.2M ImageNet images
- **Faster Training**: Converges in ~12 epochs vs. hundreds from scratch
- **Better Generalization**: Pre-trained features reduce overfitting on small medical datasets
- **Lower Data Requirements**: Effective with only 1,356 training images

### Model Configuration
```python
# Load pre-trained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Replace final layer for binary classification
model.fc = nn.Linear(512, 2)  # 512 input features → 2 classes

# Loss function with class weights (handles imbalance)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)
```

---

## 📈 Model Performance

### Test Set Metrics
| Metric | Value |
|--------|-------|
| **F1-Score** | 0.98 |
| **ROC-AUC** | 0.99 |
| **Accuracy** | 0.98 |
| **Sensitivity (Recall)** | 0.98 |
| **Specificity** | 0.99 |
| **Optimal Threshold** | 0.50 |

### Confusion Matrix (Test set example during training)
```
                Predicted
              Tumor   No Tumor
Actual Tumor    82       18      (Sensitivity: 82%)
      No Tumor  13       87      (Specificity: 87%)

### Confusion Matrix (Final)

4 FP out of 500 no-tumor cases (Sensitivity: 99.2%)
      13 FN out of 856 tumor cases (Specificity: 98.5%)
```

### Key Insights
- **High Specificity**: Low false positive rate (13%) - minimizes unnecessary alarm
- **Good Sensitivity**: Detects 82% of actual tumors
- **Balanced Performance**: F1-score of 0.83 indicates good balance between precision and recall
- **Threshold Tuning**: Optimal threshold of 0.50 selected via validation set analysis

---

## 🔍 Data Preprocessing Details

### Preprocessing Pipeline

#### 1. Duplicate Detection
- **Method**: Perceptual hashing (pHash) with Hamming distance
- **Threshold**: Distance ≤ 2 (on 0-64 scale)
- **Strategy**: Keep highest resolution image from each duplicate group
- **Result**: Removes near-identical scans (e.g., re-scans, crops)

```python
# Compute perceptual hash
hash1 = imagehash.phash(image1)
hash2 = imagehash.phash(image2)

# Calculate similarity
distance = hash1 - hash2  # Hamming distance

# Mark as duplicate if very similar
if distance <= SIMILARITY_THRESHOLD:
    mark_as_duplicate()
```

#### 2. Outlier Detection
- **Brightness Analysis**: Z-score > 5.0 standard deviations
- **Variance Check**: Pixel variance < 10 (blank images)
- **Result**: Removes corrupted, over/underexposed, or blank scans

#### 3. Image Standardization
- **Target Size**: 224×224 pixels (ResNet18 standard input)
- **Aspect Ratio**: Center-crop if width/height > 1.1
- **Interpolation**: `cv2.INTER_AREA` for high-quality downsampling

### Before vs. After Preprocessing

| Metric | Before | After |
|--------|--------|-------|
| **Total Images** | 1,500 | 1,356 |
| **Duplicates Removed** | - | 118 |
| **Outliers Removed** | - | 26 |
| **Size Standardized** | Variable | 224×224 |
| **Class Balance** | 58% / 42% | 57% / 43% |

---

## 🎨 GradCAM Interpretability

### What is GradCAM?

**Gradient-weighted Class Activation Mapping (GradCAM)** visualizes which regions of an image are most important for the model's prediction.

### How It Works

1. **Forward Pass**: Input image → Extract final convolutional layer activations
2. **Backward Pass**: Compute gradients of target class w.r.t. activations
3. **Weight Calculation**: Global average pooling of gradients
4. **Weighted Sum**: Combine activation maps using weights
5. **ReLU + Normalize**: Apply ReLU and normalize to [0, 1]
6. **Upsampling**: Resize heatmap to original image size

### Example GradCAM Outputs

```
Original Image    GradCAM Heatmap      Overlay
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   🧠 MRI    │   │   🔥 Hot     │   │  🧠 + 🔥   │
│             │ → │   Regions   │ → │  Combined   │
│             │   │   (Red)     │   │             │
└─────────────┘   └─────────────┘   └─────────────┘
```

**Red/Yellow Regions**: High importance (model focuses here)  
**Blue/Green Regions**: Low importance (model ignores)

### Validation of Model Decisions

✅ **Good**: Heatmap focuses on tumor region  
❌ **Bad**: Heatmap highlights image artifacts or borders

---

## 🌐 Web Application Features

### Single Image Upload
1. Click "Browse files" or drag-and-drop
2. Upload MRI scan (PNG, JPG, JPEG)
3. View prediction instantly
4. See confidence score and probability

### Batch Processing
1. Upload multiple images simultaneously
2. View results table with all predictions
3. Download results as CSV
4. Analyze accuracy if labels provided

### Interactive Visualizations
- **Probability Bar Chart**: Visual confidence indicator
- **Threshold Line**: Shows decision boundary (default: 0.50)
- **Model Metrics**: F1-score, ROC-AUC, confusion matrix
- **Disclaimer**: Prominent medical disclaimer

---

## 🛠️ Technical Details

### Image Normalization
All images are normalized using ImageNet statistics:
```python
mean = [0.485, 0.456, 0.406]  # RGB channels
std = [0.229, 0.224, 0.225]   # RGB channels
```

### Data Augmentation (Training Only)
```python
transforms.Compose([
    transforms.RandomHorizontalFlip(),      # 50% chance
    transforms.RandomVerticalFlip(),        # 50% chance
    transforms.RandomRotation(15),          # ±15 degrees
    transforms.ColorJitter(                 # Brightness/contrast variation
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    )
])
```

### Class Imbalance Handling
```python
# Calculate class weights inversely proportional to frequency
class_weights = [
    total_samples / (num_classes * class_count[i])
    for i in range(num_classes)
]

# Apply weighted loss
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### Device Compatibility
```python
# Automatically detect best available device
DEVICE = torch.device("mps")   if torch.backends.mps.is_available()   else \
         torch.device("cuda")  if torch.cuda.is_available()          else \
         torch.device("cpu")
```

**Supported**:
- ✅ Apple Silicon (M1/M2/M3) - Metal Performance Shaders (MPS)
- ✅ NVIDIA GPUs - CUDA
- ✅ CPU fallback

---

## 📝 Model Files

### `best_model_m1_notebook.pt`
- **Format**: PyTorch state dict (weights only)
- **Size**: ~45 MB
- **Contains**: Learned parameters (weights and biases) for all layers
- **Does NOT contain**: Model architecture, hyperparameters, preprocessing steps

### Loading the Model
```python
import torch
from torchvision import models
import torch.nn as nn

# Recreate architecture
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 2)

# Load weights
model.load_state_dict(torch.load('best_model_m1_notebook.pt'))
model.eval()

# Preprocess input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Make prediction
image = Image.open('mri_scan.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)
output = model(input_tensor)
probs = torch.nn.functional.softmax(output, dim=1)
prediction = torch.argmax(probs, dim=1).item()

print(f"Prediction: {'Tumor' if prediction == 0 else 'No Tumor'}")
print(f"Confidence: {probs[0, prediction].item():.2%}")
```

### `test_results.json`
Contains model evaluation metrics:
```json
{
  "optimal_threshold": 0.50,
  "test_f1_score": 0.8312,
  "confusion_matrix": [[82, 18], [13, 87]],
  "threshold_tuning": [
    {"threshold": 0.10, "f1_score": 0.7234},
    {"threshold": 0.50, "f1_score": 0.8312},
    {"threshold": 0.90, "f1_score": 0.6891}
  ]
}
```

---

## 🔬 Research & Citations

### Dataset
This project uses a publicly available brain MRI dataset. Please cite appropriately if using this code or dataset for research.

### Key Papers
- **ResNet**: He et al., "Deep Residual Learning for Image Recognition" (2015)
- **GradCAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (2017)
- **Transfer Learning**: Yosinski et al., "How transferable are features in deep neural networks?" (2014)

---

## 🚧 Limitations

1. **Dataset Size**: Only 1,356 images - larger datasets would improve generalization
2. **Class Imbalance**: 57% tumor / 43% non-tumor - may bias toward tumor detection
3. **Single Modality**: MRI only - does not incorporate CT, PET, or clinical data
4. **Binary Classification**: Only detects presence/absence - does not classify tumor types
5. **No Clinical Validation**: Not validated on real clinical data or by medical professionals
6. **Generalization**: Trained on specific MRI protocols - may not work on different scanners/protocols

---

## 🔮 Future Improvements

- [ ] Multi-class classification (glioma, meningioma, pituitary tumor)
- [ ] Tumor segmentation (pixel-level localization)
- [ ] Ensemble models (combine multiple architectures)
- [ ] Attention mechanisms (Transformers, Vision Transformers)
- [ ] 3D CNN support (volumetric MRI analysis)
- [ ] Clinical metadata integration (age, symptoms, history)
- [ ] Uncertainty quantification (Bayesian neural networks)
- [ ] Federated learning (train on distributed hospital data)
- [ ] DICOM support (medical imaging standard format)
- [ ] Real-time inference API (REST endpoint)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution
- Improve preprocessing pipeline
- Add new model architectures
- Enhance visualization tools
- Write unit tests
- Improve documentation
- Add multi-language support

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Medical Disclaimer

**IMPORTANT**: This software is provided for educational and research purposes only. It is NOT a medical device and is NOT intended for clinical use, medical diagnosis, or treatment decisions.

- ❌ Do NOT use for patient diagnosis
- ❌ Do NOT replace professional medical advice
- ❌ Do NOT use in clinical settings without proper validation
- ✅ Consult qualified healthcare professionals for medical decisions

The developers assume no liability for any medical decisions made using this software.

---

## 📧 Contact

**Project Maintainer**: Sean McAllister 
**Email**: sean.david.mcallister@gmail.com 
**GitHub**: https://github.com/mcallisters 


---

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- ResNet authors for the groundbreaking architecture
- GradCAM authors for model interpretability techniques
- Streamlit for the intuitive web framework
- Brain MRI dataset contributors
- Open-source community

---

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [GradCAM Paper](https://arxiv.org/abs/1610.02391)
- [Transfer Learning Guide](https://cs231n.github.io/transfer-learning/)
- [Medical Image Analysis Review](https://arxiv.org/abs/1702.05747)

---

**Built with ❤️ for advancing medical AI research**