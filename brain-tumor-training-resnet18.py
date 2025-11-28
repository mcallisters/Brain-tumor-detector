import os
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt

# 2️⃣ Device configuration
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {DEVICE}")

# 3️⃣ Configuration
DATA_DIR = '/Users/seanmcallister/Data_Science/SuperDataScientist/SDS-CP041-neuroscan/advanced/submissions/team-members/mcallister/brain_tumor_data_preprocessed_all'
IMG_SIZE = 224
NUM_CLASSES = 2
BATCH_SIZE = 16
NUM_EPOCHS = 12
LEARNING_RATE = 1e-4
PATIENCE = 5
CLASSES = ['yes', 'no']

# 4️⃣ Custom Dataset Class for Loading Brain Tumor Images
class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, classes=CLASSES, transform=None):
        """
        Initializes the dataset by scanning directories for images.
        
        Args:
            data_dir: Root directory containing subdirectories for each class
            classes: List of class names (e.g., ['yes', 'no'])
            transform: PyTorch transforms to apply to images
        """
        self.data = []
        self.labels = []
        self.transform = transform
        
        for idx, cls in enumerate(classes):
            class_dir = os.path.join(data_dir, cls)
            for img_path in glob(os.path.join(class_dir, '*.*')):
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append(img_path)
                    self.labels.append(idx)
        
        assert len(self.data) > 0, "No images found!"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 5️⃣ Image Preprocessing - Training Transforms (with data augmentation)
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 5️⃣ Image Preprocessing - Validation/Test Transforms (minimal, no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 6️⃣ Train/Validation/Test Split (70/20/10)
all_data = BrainTumorDataset(DATA_DIR, transform=train_transform)

# First split: 70% training, 30% temp (validation + test)
train_indices, temp_indices = train_test_split(
    np.arange(len(all_data)),
    test_size=0.30,
    stratify=all_data.labels,
    random_state=42
)

# Second split: Split the 30% into 20% validation and 10% test
# 10/30 = 0.333 to get 10% of total from the 30%
val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.333,
    stratify=[all_data.labels[i] for i in temp_indices],
    random_state=42
)

print(f"\n{'='*80}")
print("DATA SPLIT SUMMARY")
print(f"{'='*80}")
print(f"Total images: {len(all_data)}")
print(f"Training set: {len(train_indices)} images ({len(train_indices)/len(all_data)*100:.1f}%)")
print(f"Validation set: {len(val_indices)} images ({len(val_indices)/len(all_data)*100:.1f}%)")
print(f"Test set: {len(test_indices)} images ({len(test_indices)/len(all_data)*100:.1f}%)")
print(f"{'='*80}\n")

# Create training subset with training transforms (includes augmentation)
train_dataset = Subset(all_data, train_indices)

# Create validation subset with validation transforms (no augmentation)
val_dataset = Subset(
    BrainTumorDataset(DATA_DIR, transform=val_transform),
    val_indices
)

# Create test subset with validation transforms (no augmentation)
test_dataset = Subset(
    BrainTumorDataset(DATA_DIR, transform=val_transform),
    test_indices
)

# 7️⃣ DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# 8️⃣ Class Weights - Handle Imbalanced Dataset
train_labels = [all_data.labels[i] for i in train_indices]
label_counts = Counter(train_labels)

total = sum(label_counts.values())

class_weights = [total / (NUM_CLASSES * label_counts[i]) for i in range(NUM_CLASSES)]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f"Class weights: {class_weights_tensor}")

# 9️⃣ Model Setup
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 🔹 Training Loop
best_val_f1 = 0
patience_counter = 0
YES_CLASS_IDX = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total_train = 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total_train += labels.size(0)
    
    train_loss = running_loss / total_train
    train_acc = correct / total_train
    
    # Validation Phase
    model.eval()
    val_labels_list = []
    val_probs_yes = []
    val_preds_list = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = nn.functional.softmax(outputs, dim=1)
            predicted = torch.argmax(probs, dim=1)
            
            val_labels_list.extend(labels.cpu().numpy())
            val_preds_list.extend(predicted.cpu().numpy())
            val_probs_yes.extend(probs[:, YES_CLASS_IDX].cpu().numpy())
    
    binary_labels = np.array([1 if l == YES_CLASS_IDX else 0 for l in val_labels_list])
    binary_preds = np.array([1 if p >= 0.5 else 0 for p in val_probs_yes])
    
    val_f1_yes = f1_score(binary_labels, binary_preds)
    cm = confusion_matrix(val_labels_list, val_preds_list)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val F1 (yes): {val_f1_yes:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    if val_f1_yes > best_val_f1:
        best_val_f1 = val_f1_yes
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model_m1_notebook.pt')
        print(f"✓ Model saved! New best F1: {best_val_f1:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered!")
            break

# 🔹 Threshold Tuning on Validation Set
model.eval()
all_val_labels = []
all_val_probs = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        probs = nn.functional.softmax(outputs, dim=1)[:, YES_CLASS_IDX]
        all_val_labels.extend(labels.cpu().numpy())
        all_val_probs.extend(probs.cpu().numpy())

binary_val_labels = np.array([1 if l == YES_CLASS_IDX else 0 for l in all_val_labels])

thresholds = np.arange(0.1, 0.91, 0.05)
best_thresh = 0.5
best_f1_yes = 0
threshold_results = []

for t in thresholds:
    preds = np.array([1 if p >= t else 0 for p in all_val_probs])
    f1_yes = f1_score(binary_val_labels, preds)
    threshold_results.append({'threshold': t, 'f1_score': f1_yes})
    if f1_yes > best_f1_yes:
        best_f1_yes = f1_yes
        best_thresh = t

print(f"\n{'='*80}")
print("THRESHOLD TUNING RESULTS (on Validation Set)")
print(f"{'='*80}")
for result in threshold_results:
    print(f"Threshold: {result['threshold']:.2f}, F1-score: {result['f1_score']:.4f}")
print(f"Optimal threshold: {best_thresh:.2f}, F1-score: {best_f1_yes:.4f}")
print(f"{'='*80}\n")

# 🔹 Test Set Evaluation
print(f"{'='*80}")
print("TEST SET EVALUATION (Final Metrics)")
print(f"{'='*80}\n")

test_labels_list = []
test_probs_yes = []
test_preds_list = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        probs = nn.functional.softmax(outputs, dim=1)
        predicted = torch.argmax(probs, dim=1)
        
        test_labels_list.extend(labels.cpu().numpy())
        test_preds_list.extend(predicted.cpu().numpy())
        test_probs_yes.extend(probs[:, YES_CLASS_IDX].cpu().numpy())

binary_test_labels = np.array([1 if l == YES_CLASS_IDX else 0 for l in test_labels_list])
binary_test_preds = np.array([1 if p >= best_thresh else 0 for p in test_probs_yes])

test_f1_yes = f1_score(binary_test_labels, binary_test_preds)
test_cm = confusion_matrix(test_labels_list, test_preds_list)

print(f"Test Set F1 (yes class): {test_f1_yes:.4f}")
print(f"Test Set Confusion Matrix:\n{test_cm}")
print(f"Optimal threshold used: {best_thresh:.2f}")
print(f"{'='*80}\n")

# Save test results for reference
import json
test_results = {
    'optimal_threshold': float(best_thresh),
    'test_f1_score': float(test_f1_yes),
    'confusion_matrix': test_cm.tolist(),
    'threshold_tuning': threshold_results
}
with open('test_results.json', 'w') as f:
    json.dump(test_results, f, indent=2)
print("Test results saved to test_results.json")