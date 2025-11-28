import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob
import random

# Configuration
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
MODEL_PATH = 'best_model_m1_notebook.pt'
DATA_DIR = '/Users/seanmcallister/Data_Science/SuperDataScientist/SDS-CP041-neuroscan/advanced/submissions/team-members/mcallister/brain_tumor_data_preprocessed_all'
IMG_SIZE = 224
NUM_CLASSES = 2
CLASSES = ['yes', 'no']  # yes=tumor, no=no tumor
YES_CLASS_IDX = 0
OUTPUT_DIR = '/Users/seanmcallister/Data_Science/SuperDataScientist/SDS-CP041-neuroscan/advanced/submissions/team-members/mcallister/gradcam_visualizations'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image preprocessing (same as validation transform)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Denormalization for visualization
def denormalize(tensor):
    """Convert normalized tensor back to original image for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean


class GradCAM:
    """GradCAM implementation for visualizing CNN decisions"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to capture forward pass activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to capture backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class):
        """Generate GradCAM heatmap for a specific class"""
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Get prediction and probability
        probs = nn.functional.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
        
        # Backward pass for target class
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # Apply ReLU (only positive contributions)
        cam = nn.functional.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input image size
        cam = nn.functional.interpolate(
            cam, 
            size=(IMG_SIZE, IMG_SIZE), 
            mode='bilinear', 
            align_corners=False
        )
        
        cam = cam.squeeze().cpu().numpy()
        
        return cam, pred_class, confidence
    
    def remove_hooks(self):
        """Clean up hooks"""
        self.target_layer._forward_hooks.clear()
        self.target_layer._backward_hooks.clear()


def apply_colormap(heatmap, colormap=cv2.COLORMAP_JET):
    """Apply colormap to heatmap"""
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    return colored_heatmap / 255.0


def overlay_heatmap(image, heatmap, alpha=0.4):
    """Overlay heatmap on original image"""
    colored_heatmap = apply_colormap(heatmap)
    overlayed = colored_heatmap * alpha + image * (1 - alpha)
    return np.clip(overlayed, 0, 1)


def visualize_gradcam(image_path, model, gradcam, save_path=None):
    """Generate and visualize GradCAM for a single image"""
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(original_image).unsqueeze(0).to(DEVICE)
    
    # Generate GradCAM for tumor class (YES_CLASS_IDX)
    heatmap, pred_class, confidence = gradcam.generate_cam(input_tensor, YES_CLASS_IDX)
    
    # Prepare original image for visualization
    img_np = np.array(original_image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    
    # Create overlay
    overlayed_image = overlay_heatmap(img_np, heatmap)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap only
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('GradCAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlayed_image)
    pred_label = CLASSES[pred_class]
    true_label = 'yes' if 'yes' in image_path.lower() else 'no'
    title = f'Prediction: {pred_label} ({confidence:.2%})\nTrue Label: {true_label}'
    color = 'green' if pred_label == true_label else 'red'
    axes[2].set_title(title, fontsize=12, fontweight='bold', color=color)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    
    return pred_class, confidence, true_label


def batch_visualize(data_dir, model, gradcam, num_samples=10, classes=['yes', 'no']):
    """Generate GradCAM visualizations for multiple images from each class"""
    
    print(f"\n{'='*80}")
    print("GENERATING GRADCAM VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    for cls in classes:
        print(f"\nProcessing class: {cls}")
        class_dir = os.path.join(data_dir, cls)
        all_images = glob(os.path.join(class_dir, '*.*'))
        # Randomly sample images
        image_paths = random.sample(all_images, min(num_samples, len(all_images)))
        
        for idx, img_path in enumerate(image_paths):
            # Create save path
            filename = os.path.basename(img_path).split('.')[0]
            save_path = os.path.join(OUTPUT_DIR, f'{cls}_{idx+1}_{filename}_gradcam.png')
            
            # Generate visualization
            try:
                pred_class, confidence, true_label = visualize_gradcam(
                    img_path, model, gradcam, save_path
                )
                status = "✓" if CLASSES[pred_class] == true_label else "✗"
                print(f"  {status} Image {idx+1}: Pred={CLASSES[pred_class]} ({confidence:.2%}), True={true_label}")
            except Exception as e:
                print(f"  ✗ Error processing {img_path}: {str(e)}")
    
    print(f"\n{'='*80}")
    print(f"Visualizations saved to: {OUTPUT_DIR}/")
    print(f"{'='*80}\n")


def analyze_single_image(image_path, model, gradcam):
    """Interactive visualization for a single image"""
    print(f"\nAnalyzing: {image_path}")
    
    save_path = os.path.join(OUTPUT_DIR, f'single_analysis_{os.path.basename(image_path)}.png')
    pred_class, confidence, true_label = visualize_gradcam(
        image_path, model, gradcam, save_path
    )
    
    print(f"Prediction: {CLASSES[pred_class]} (confidence: {confidence:.2%})")
    print(f"True Label: {true_label}")
    print(f"Result: {'Correct ✓' if CLASSES[pred_class] == true_label else 'Incorrect ✗'}")


# Main execution
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # Load trained model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Initialize GradCAM (target ResNet18's layer4 - last conv block)
    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)
    print("✓ GradCAM initialized on layer4 (last convolutional block)")
    
    # Option 1: Batch visualization (sample images from each class)
    print("\n" + "="*80)
    print("OPTION 1: BATCH VISUALIZATION")
    print("="*80)
    batch_visualize(DATA_DIR, model, gradcam, num_samples=5, classes=CLASSES)
    
    # Option 2: Analyze specific image (uncomment and modify path as needed)
    # print("\n" + "="*80)
    # print("OPTION 2: SINGLE IMAGE ANALYSIS")
    # print("="*80)
    # specific_image = '/path/to/your/specific/image.jpg'
    # analyze_single_image(specific_image, model, gradcam)
    
    # Clean up
    gradcam.remove_hooks()
    print("\n✓ GradCAM analysis complete!")