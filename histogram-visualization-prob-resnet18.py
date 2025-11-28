import os
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

# ========== CONFIGURATION ==========
MODEL_PATH = '/Users/seanmcallister/Data_Science/SuperDataScientist/SDS-CP041-neuroscan/advanced/submissions/team-members/mcallister/best_model_m1_notebook.pt'
DATA_DIR = '/Users/seanmcallister/Data_Science/SuperDataScientist/SDS-CP041-neuroscan/advanced/submissions/team-members/mcallister/brain_tumor_data_preprocessed_all'

IMG_SIZE = 224
NUM_CLASSES = 2
CLASSES = ['yes', 'no']
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
OPTIMAL_THRESHOLD = 0.5
BATCH_SIZE = 32

print(f"Using device: {DEVICE}")

# ========== DATASET CLASS ==========
class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, classes=CLASSES, transform=None):
        self.data = []
        self.labels = []
        self.label_names = []
        self.transform = transform
        for idx, cls in enumerate(classes):
            class_dir = os.path.join(data_dir, cls)
            for img_path in glob(os.path.join(class_dir, '*.*')):
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append(img_path)
                    self.labels.append(idx)
                    self.label_names.append(cls)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        label_name = self.label_names[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, label_name, img_path

# ========== LOAD MODEL ==========
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print(f"Model loaded from {MODEL_PATH}")

# ========== TRANSFORMS ==========
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ========== LOAD DATA ==========
dataset = BrainTumorDataset(DATA_DIR, transform=val_transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Total images: {len(dataset)}")

# ========== INFERENCE ==========
all_images = []
all_labels = []
all_label_names = []
all_probs_yes = []
all_predictions = []
all_img_paths = []

with torch.no_grad():
    for images, labels, label_names, img_paths in dataloader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = nn.functional.softmax(outputs, dim=1)
        probs_yes = probs[:, 0].cpu().numpy()
        
        predictions = np.array([1 if p >= OPTIMAL_THRESHOLD else 0 for p in probs_yes])
        
        all_images.extend(images.cpu())
        all_labels.extend(labels.numpy())
        all_label_names.extend(label_names)
        all_probs_yes.extend(probs_yes)
        all_predictions.extend(predictions)
        all_img_paths.extend(img_paths)

all_images = np.array(all_images)
all_labels = np.array(all_labels)
all_probs_yes = np.array(all_probs_yes)
all_predictions = np.array(all_predictions)

print(f"Inference complete. Total predictions: {len(all_predictions)}")

# ========== CATEGORIZE PREDICTIONS ==========
tp_idx = np.where((all_labels == 0) & (all_predictions == 1))[0]
tn_idx = np.where((all_labels == 1) & (all_predictions == 0))[0]
fp_idx = np.where((all_labels == 1) & (all_predictions == 1))[0]
fn_idx = np.where((all_labels == 0) & (all_predictions == 0))[0]

print(f"\n{'='*50}")
print(f"CONFUSION MATRIX BREAKDOWN")
print(f"{'='*50}")
print(f"True Positives (tumor correctly detected): {len(tp_idx)}")
print(f"True Negatives (no tumor correctly identified): {len(tn_idx)}")
print(f"False Positives (false alarm): {len(fp_idx)}")
print(f"False Negatives (missed tumor): {len(fn_idx)}")
print(f"{'='*50}\n")

# ========== VISUALIZATION FUNCTION ==========
def visualize_category(indices, category_name, num_samples=12):
    """Visualize images from a specific category"""
    if len(indices) == 0:
        print(f"No {category_name} found!")
        return
    
    sample_indices = np.random.choice(indices, size=min(num_samples, len(indices)), replace=False)
    
    n_samples = len(sample_indices)
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        
        img = all_images[idx]
        img = img.transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        prob_yes = all_probs_yes[idx]
        true_label = all_label_names[idx]
        pred_label = 'yes' if prob_yes >= OPTIMAL_THRESHOLD else 'no'
        
        is_correct = (all_labels[idx] == 0 and pred_label == 'yes') or \
                     (all_labels[idx] == 1 and pred_label == 'no')
        color = 'green' if is_correct else 'red'
        
        title = f"True: {true_label} | Pred: {pred_label}\nProb(yes): {prob_yes:.3f}"
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
        ax.axis('off')
    
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"{category_name} (Total: {len(indices)})", y=1.00, fontsize=14, fontweight='bold')
    plt.show()

# ========== PROBABILITY DISTRIBUTION ==========
def plot_distributions():
    """Plot probability distributions by true label"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    yes_probs = all_probs_yes[all_labels == 0]
    axes[0].hist(yes_probs, bins=20, color='#FF6B6B', alpha=0.7, edgecolor='black')
    axes[0].axvline(OPTIMAL_THRESHOLD, color='black', linestyle='--', linewidth=2, label=f'Threshold: {OPTIMAL_THRESHOLD}')
    axes[0].set_xlabel("Predicted Probability of Tumor", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title(f"Tumor Images (n={len(yes_probs)})", fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    no_probs = all_probs_yes[all_labels == 1]
    axes[1].hist(no_probs, bins=20, color='#4ECDC4', alpha=0.7, edgecolor='black')
    axes[1].axvline(OPTIMAL_THRESHOLD, color='black', linestyle='--', linewidth=2, label=f'Threshold: {OPTIMAL_THRESHOLD}')
    axes[1].set_xlabel("Predicted Probability of Tumor", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title(f"Non-Tumor Images (n={len(no_probs)})", fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ========== ROC CURVE ==========
def plot_roc_curve():
    """Plot ROC curve"""
    # Convert labels to binary (1 for tumor, 0 for non-tumor)
    binary_labels = np.array([1 if label == 0 else 0 for label in all_labels])
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(binary_labels, all_probs_yes)
    roc_auc = auc(fpr, tpr)
    
    # Calculate FPR and TPR at our optimal threshold
    preds_at_threshold = np.array([1 if p >= OPTIMAL_THRESHOLD else 0 for p in all_probs_yes])
    tn = np.sum((binary_labels == 0) & (preds_at_threshold == 0))
    fp = np.sum((binary_labels == 0) & (preds_at_threshold == 1))
    fn = np.sum((binary_labels == 1) & (preds_at_threshold == 0))
    tp = np.sum((binary_labels == 1) & (preds_at_threshold == 1))
    
    fpr_at_threshold = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr_at_threshold = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    plt.figure(figsize=(10, 7))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='#FF6B6B', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    
    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
    
    # Mark current threshold
    plt.plot(fpr_at_threshold, tpr_at_threshold, 'go', markersize=12, label=f'Current Threshold ({OPTIMAL_THRESHOLD})\nFPR: {fpr_at_threshold:.3f}, TPR: {tpr_at_threshold:.3f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curve - Brain Tumor Detection', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ========== AUTO-GENERATE ALL VISUALIZATIONS ==========
print("Generating visualizations...\n")

print("📊 PROBABILITY DISTRIBUTIONS")
plot_distributions()

print("📈 ROC CURVE")
plot_roc_curve()

print("✅ TRUE POSITIVES (Detected Tumors)")
visualize_category(tp_idx, "TRUE POSITIVES (Detected Tumors)", num_samples=12)

print("✅ TRUE NEGATIVES (Correct Non-Tumors)")
visualize_category(tn_idx, "TRUE NEGATIVES (Correct Non-Tumors)", num_samples=12)

print("⚠️  FALSE POSITIVES (False Alarms)")
visualize_category(fp_idx, "FALSE POSITIVES (False Alarms)", num_samples=24)

print("🚨 FALSE NEGATIVES (Missed Tumors)")
visualize_category(fn_idx, "FALSE NEGATIVES (Missed Tumors)", num_samples=12)

print("\n✨ All visualizations complete!")