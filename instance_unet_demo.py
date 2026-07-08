import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import cv2
from scipy import ndimage

# Import from main file
from unet_model import InstanceUNet, download_oxford_pet_dataset, process_instances

def print_model_summary(model):
    """Print model architecture summary to verify structure"""
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} total parameters")
    
    # Print layer structure
    for name, module in model.named_children():
        print(f"Layer: {name}, Type: {type(module).__name__}")
        
    # Print a sample of weights to ensure they're loaded
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter {name}: shape {param.shape}")
            print(f"  - Min: {param.min().item():.4f}, Max: {param.max().item():.4f}")
            print(f"  - Non-zero values: {torch.sum(param != 0).item()}/{param.numel()}")
            break  # Just print one parameter as a sample

def load_model(model_path, device=None):
    """Load a trained Instance U-Net model with enhanced error checking"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model on {device}...")
    
    # Initialize model
    model = InstanceUNet(n_channels=3, n_classes=1, embedding_dim=8, bilinear=True)
    
    # Load weights with extensive error checking
    if os.path.isfile(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            print(f"Checkpoint loaded, type: {type(checkpoint)}")
            
            # Check if the checkpoint contains state_dict directly or within a dictionary
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                print("Loading from checkpoint dictionary...")
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Print epoch and loss info if available
                if 'epoch' in checkpoint:
                    print(f"Checkpoint from epoch: {checkpoint['epoch']+1}")
                if 'train_loss' in checkpoint:
                    print(f"Training loss: {checkpoint['train_loss']:.6f}")
                if 'val_loss' in checkpoint:
                    print(f"Validation loss: {checkpoint['val_loss']:.6f}")
            else:
                print("Loading direct state dictionary...")
                model.load_state_dict(checkpoint)
                
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    else:
        raise FileNotFoundError(f"No model found at {model_path}")
    
    model.to(device)
    model.eval()
    
    # Print model summary to verify correct loading
    print_model_summary(model)
    
    return model, device

def preprocess_image(image_path, size=(256, 256)):
    """Preprocess an image for prediction"""
    # Load image with error handling
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        raise
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    
    # Apply transform
    image_tensor = transform(image)
    
    # Verify tensor properties
    print(f"Image tensor shape: {image_tensor.shape}")
    print(f"Image tensor range: min={image_tensor.min().item():.4f}, max={image_tensor.max().item():.4f}")
    
    return image_tensor, image

def predict_instances(model, image_tensor, device):
    """Generate instance segmentation for an image"""
    # Ensure model is in evaluation mode
    model.eval()
    
    # Move image to device
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Generate prediction
    with torch.no_grad():
        semantic_logits, center_logits, embeddings = model(image_tensor)
        
        # Use lower threshold for semantic segmentation
        semantic_mask = torch.sigmoid(semantic_logits) > 0.002
        center_mask = torch.sigmoid(center_logits) > 0.002
        
        # Print statistics about raw predictions
        print(f"Semantic prediction: Min={torch.sigmoid(semantic_logits).min().item():.6f}, "
              f"Max={torch.sigmoid(semantic_logits).max().item():.6f}")
        print(f"Center prediction: Min={torch.sigmoid(center_logits).min().item():.6f}, "
              f"Max={torch.sigmoid(center_logits).max().item():.6f}")
    
    # Convert to numpy arrays
    semantic_mask = semantic_mask.squeeze().cpu().numpy()
    center_mask = center_mask.squeeze().cpu().numpy()
    embeddings = embeddings.squeeze().cpu().numpy()
    
    # Process instance segmentation
    colored_instances, instance_masks = process_instances(semantic_mask, center_mask, embeddings)
    
    return semantic_mask, center_mask, embeddings, colored_instances, instance_masks

def overlay_instances(image, colored_instances, alpha=0.5):
    """Overlay instance segmentation on original image"""
    # Ensure image is numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Resize colored instances to match image dimensions
    colored_instances_resized = cv2.resize(colored_instances, (image.shape[1], image.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
    
    # Create overlay
    overlay = cv2.addWeighted(image, 1-alpha, colored_instances_resized, alpha, 0)
    
    return overlay

def process_and_visualize(image_path, model, device, output_dir='results'):
    """Process an image and visualize instance segmentation results"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Preprocess image
    image_tensor, original_image = preprocess_image(image_path)
    
    # Generate instance segmentation
    semantic_mask, center_mask, embeddings, colored_instances, instance_masks = predict_instances(
        model, image_tensor, device)
    
    # Convert original image for visualization
    original_image_np = np.array(original_image)
    
    # Create overlaid image
    overlaid_image = overlay_instances(original_image_np.copy(), colored_instances, alpha=0.5)
    
    # Create visualization figure
    plt.figure(figsize=(16, 12))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(original_image_np)
    plt.title('Original Image')
    plt.axis('off')
    
    # Semantic segmentation
    plt.subplot(2, 3, 2)
    plt.imshow(semantic_mask, cmap='gray')
    plt.title('Semantic Segmentation')
    plt.axis('off')
    
    # Center predictions
    plt.subplot(2, 3, 3)
    plt.imshow(center_mask, cmap='jet')
    plt.title('Center Predictions')
    plt.axis('off')
    
    # Embeddings visualization (first 3 channels as RGB)
    if embeddings.shape[0] >= 3:
        embedding_vis = np.stack([
            embeddings[0],
            embeddings[1],
            embeddings[2]
        ], axis=-1)
        
        # Normalize for visualization
        if embedding_vis.max() > embedding_vis.min():
            embedding_vis = (embedding_vis - embedding_vis.min()) / (embedding_vis.max() - embedding_vis.min())
        
        # Zero out background
        embedding_vis = embedding_vis * semantic_mask.reshape(*semantic_mask.shape, 1)
        
        plt.subplot(2, 3, 4)
        plt.imshow(embedding_vis)
        plt.title('Embedding Features (RGB)')
        plt.axis('off')
    else:
        plt.subplot(2, 3, 4)
        plt.text(0.5, 0.5, "Not enough embedding channels for RGB visualization", 
                 ha='center', va='center')
        plt.axis('off')
    
    # Instance segmentation (colored)
    plt.subplot(2, 3, 5)
    plt.imshow(colored_instances)
    plt.title(f'Instance Segmentation ({len(instance_masks)} instances)')
    plt.axis('off')
    
    # Overlay on original image
    plt.subplot(2, 3, 6)
    plt.imshow(overlaid_image)
    plt.title('Instances Overlay')
    plt.axis('off')
    
    # Save visualization
    output_path = os.path.join(output_dir, f'{base_filename}_instances.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Results saved to {output_path}")
    
    # Separately visualize each instance
    if len(instance_masks) > 1:
        plt.figure(figsize=(15, 5 * len(instance_masks)))
        plt.suptitle(f"Individual Instances for {base_filename}", fontsize=16)
        
        for i, mask in enumerate(instance_masks):
            # Create colored mask for this instance
            color = np.array([
                (i * 35) % 256,
                (i * 60) % 256,
                (i * 111) % 256
            ], dtype=np.uint8)
            
            instance_colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
            instance_colored[mask] = color
            
            # Create overlay for this instance
            instance_overlay = overlay_instances(original_image_np.copy(), instance_colored, alpha=0.5)
            
            # Plot mask
            plt.subplot(len(instance_masks), 3, i*3 + 1)
            plt.imshow(mask, cmap='gray')
            plt.title(f'Instance {i+1} Mask')
            plt.axis('off')
            
            # Plot colored instance
            plt.subplot(len(instance_masks), 3, i*3 + 2)
            plt.imshow(instance_colored)
            plt.title(f'Instance {i+1} Colored')
            plt.axis('off')
            
            # Plot overlay
            plt.subplot(len(instance_masks), 3, i*3 + 3)
            plt.imshow(instance_overlay)
            plt.title(f'Instance {i+1} Overlay')
            plt.axis('off')
        
        # Save individual instances visualization
        instances_path = os.path.join(output_dir, f'{base_filename}_individual_instances.png')
        plt.tight_layout()
        plt.savefig(instances_path, dpi=150)
        plt.close()
        
        print(f"Individual instances saved to {instances_path}")
    
    return original_image_np, semantic_mask, colored_instances, overlaid_image, instance_masks

def run_demo(model_path='instance_unet_model_final.pth', data_dir='./data', num_samples=5):
    """Run demonstration on sample images from the dataset"""
    print("\n===== Instance U-Net Segmentation Demo =====\n")
    
    # Load model with enhanced verification
    try:
        model, device = load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Get dataset directories
    print("\n===== Processing Sample Images =====\n")
    images_dir, masks_dir = download_oxford_pet_dataset(data_dir)
    
    # Get list of image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Select random samples
    if len(image_files) > num_samples:
        selected_images = np.random.choice(image_files, num_samples, replace=False)
    else:
        selected_images = image_files[:num_samples]
    
    # Process each image
    results = []
    for image_file in selected_images:
        image_path = os.path.join(images_dir, image_file)
        print(f"\nProcessing {image_file}...")
        
        # Process image and get results
        original, semantic_mask, colored_instances, overlaid, instance_masks = process_and_visualize(
            image_path, model, device)
        
        results.append((original, semantic_mask, colored_instances, overlaid, image_file, len(instance_masks)))
    
    # Create a combined visualization of all results
    plt.figure(figsize=(16, 6 * num_samples))
    
    for i, (original, semantic, instances, overlaid, filename, num_instances) in enumerate(results):
        # Display original image
        plt.subplot(num_samples, 4, i*4 + 1)
        plt.imshow(original)
        plt.title(f'Original: {filename}')
        plt.axis('off')
        
        # Display semantic mask
        plt.subplot(num_samples, 4, i*4 + 2)
        plt.imshow(semantic, cmap='gray')
        plt.title('Semantic Segmentation')
        plt.axis('off')
        
        # Display instance segmentation
        plt.subplot(num_samples, 4, i*4 + 3)
        plt.imshow(instances)
        plt.title(f'Instance Segmentation ({num_instances})')
        plt.axis('off')
        
        # Display overlay
        plt.subplot(num_samples, 4, i*4 + 4)
        plt.imshow(overlaid)
        plt.title('Instances Overlay')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'all_instance_results.png'), dpi=150)
    plt.close()
    
    print("\n===== Demo Complete =====")
    print(f"Processed {len(results)} images successfully!")
    print(f"All results saved in the 'results' directory.")

if __name__ == "__main__":
    # Check if model exists, otherwise notify to train first
    if os.path.exists('instance_unet_model_final.pth') or os.path.exists('instance_unet_checkpoint_epoch_10.pth'):
        print("Using existing trained model...")
        # Try final model first, fall back to last checkpoint
        if os.path.exists('instance_unet_model_final.pth'):
            run_demo('instance_unet_model_final.pth')
        else:
            run_demo('instance_unet_checkpoint_epoch_10.pth')
    else:
        # Look for any checkpoint
        checkpoints = [f for f in os.listdir('.') if f.startswith('instance_unet_checkpoint_epoch_')]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints)[-1]
            print(f"Using latest checkpoint: {latest_checkpoint}")
            run_demo(latest_checkpoint)
        else:
            print("No trained model found. Please run unet_model.py first to train the model.")
