import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss, MSELoss
import requests
import tarfile
from sklearn.model_selection import train_test_split
import cv2
from scipy import ndimage

class DoubleConv(nn.Module):
    """Double Convolution block with batch normalization"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downsampling with maxpool followed by double convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upsampling followed by double convolution"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # Upsampling layer options
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Padding if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Final output convolution layer"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class InstanceUNet(nn.Module):
    """Enhanced U-Net architecture for instance segmentation"""
    def __init__(self, n_channels, n_classes, embedding_dim=8, bilinear=True):
        super(InstanceUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.bilinear = bilinear
        
        factor = 2 if bilinear else 1
        
        # Encoder path
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder path
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Semantic segmentation head
        self.outc = OutConv(64, n_classes)
        
        # Center prediction head (for instance centers)
        self.center_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
        # Instance embedding head (for instance discrimination)
        self.embedding_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, embedding_dim, kernel_size=1)
        )
    
    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        features = self.up4(x, x1)
        
        # Three output heads
        semantic_logits = self.outc(features)
        center_logits = self.center_head(features)
        embeddings = self.embedding_head(features)
        
        return semantic_logits, center_logits, embeddings

class OxfordPetDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None):
        """
        Dataset for Oxford Pet dataset
        
        Args:
            images_dir: Directory with all the images
            masks_dir: Directory with all the masks
            transform: Transform to be applied on images
            mask_transform: Transform to be applied on masks
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Get mask filename - in Oxford Pet dataset, masks are in trimap format
        mask_name = img_name.split('.')[0] + '.png'
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Load and process the mask correctly
        try:
            mask = Image.open(mask_path)
            mask_np = np.array(mask)
            
            # Create binary mask where only class 1 (pet) is foreground
            # In Oxford Pet dataset:
            # 1 = pet, 2 = background, 3 = boundary
            binary_mask = np.zeros_like(mask_np)
            binary_mask[mask_np == 1] = 1  # Only the pet is foreground
            
            # Create center mask (assuming center is 1/3 of the pet area)
            distance = ndimage.distance_transform_edt(binary_mask)
            max_dist = distance.max() if distance.max() > 0 else 1
            center_mask = (distance > (0.6 * max_dist)).astype(np.float32)
            
            mask = Image.fromarray(binary_mask.astype(np.uint8))
            center_mask = Image.fromarray(center_mask.astype(np.uint8))
        except Exception as e:
            print(f"Error processing mask for {img_name}: {e}")
            mask = Image.new("L", image.size, 0)
            center_mask = Image.new("L", image.size, 0)
        
        # Apply image transformations
        if self.transform:
            image = self.transform(image)
        
        # Apply mask transformations
        if self.mask_transform:
            mask = self.mask_transform(mask)
            center_mask = self.mask_transform(center_mask)
        else:
            # Default mask transform
            mask = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)(mask)
            mask = transforms.ToTensor()(mask)
            
            center_mask = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)(center_mask)
            center_mask = transforms.ToTensor()(center_mask)
        
        return image, mask, center_mask

def download_oxford_pet_dataset(data_dir='./data'):
    """Function to download and extract the Oxford Pet dataset"""
    os.makedirs(data_dir, exist_ok=True)
    
    # Download images
    images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    images_path = os.path.join(data_dir, "images.tar.gz")
    
    # Download annotations (including segmentation masks)
    annotations_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    annotations_path = os.path.join(data_dir, "annotations.tar.gz")
    
    # Download files if they don't exist
    if not os.path.exists(images_path):
        print("Downloading images...")
        response = requests.get(images_url, stream=True)
        with open(images_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    
    if not os.path.exists(annotations_path):
        print("Downloading annotations...")
        response = requests.get(annotations_url, stream=True)
        with open(annotations_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    
    # Extract files
    if not os.path.exists(os.path.join(data_dir, "images")):
        print("Extracting images...")
        with tarfile.open(images_path, 'r:gz') as tar:
            tar.extractall(path=data_dir)
    
    if not os.path.exists(os.path.join(data_dir, "annotations")):
        print("Extracting annotations...")
        with tarfile.open(annotations_path, 'r:gz') as tar:
            tar.extractall(path=data_dir)
    
    print("Dataset downloaded and extracted successfully!")
    
    # Return paths to images and masks
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "annotations", "trimaps")
    
    return images_dir, masks_dir

def visualize_sample(image, mask, center=None, title="Sample Data Visualization"):
    """Visualize a sample image and its mask"""
    if center is None:
        plt.figure(figsize=(12, 6))
        num_cols = 3
    else:
        plt.figure(figsize=(16, 6))
        num_cols = 4
    
    # Display image
    plt.subplot(1, num_cols, 1)
    plt.title("Original Image")
    plt.imshow(image.permute(1, 2, 0))  # Convert from (C,H,W) to (H,W,C)
    plt.axis('off')
    
    # Display mask
    plt.subplot(1, num_cols, 2)
    plt.title("Segmentation Mask")
    plt.imshow(mask.squeeze(), cmap='gray')  # Squeeze to remove channel dimension
    plt.axis('off')
    
    # Display center mask if provided
    if center is not None:
        plt.subplot(1, num_cols, 3)
        plt.title("Center Mask")
        plt.imshow(center.squeeze(), cmap='jet')
        plt.axis('off')
        col_offset = 1
    else:
        col_offset = 0
    
    # Display overlay
    plt.subplot(1, num_cols, 3 + col_offset)
    plt.title("Overlay")
    
    # Create RGB mask for overlay (red channel)
    rgb_mask = torch.zeros((3, mask.shape[1], mask.shape[2]))
    rgb_mask[0] = mask.squeeze() * 1.0  # Red channel
    
    # Combine image with mask
    alpha = 0.5
    overlaid = image * (1 - alpha) + rgb_mask * alpha
    
    plt.imshow(overlaid.permute(1, 2, 0))
    plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def train_instance_unet(data_dir='./data', batch_size=4, epochs=10, learning_rate=0.001):
    """Main function to train Instance U-Net model on Oxford Pet dataset"""
    # Download dataset and get directories
    images_dir, masks_dir = download_oxford_pet_dataset(data_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define transformations for images
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Define transformations for masks
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    
    # Create dataset
    dataset = OxfordPetDataset(
        images_dir=images_dir, 
        masks_dir=masks_dir, 
        transform=image_transform,
        mask_transform=mask_transform
    )
    
    # Visualize a few samples to verify correct mask processing
    print("Visualizing sample data...")
    for i in range(3):  # Check 3 random samples
        image, mask, center = dataset[i]
        visualize_sample(image, mask, center, f"Sample {i+1}")
    
    # Split dataset into train and validation sets
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Initialize model
    model = InstanceUNet(n_channels=3, n_classes=1, embedding_dim=8, bilinear=True).to(device)
    
    # Define loss functions and optimizer
    semantic_criterion = BCEWithLogitsLoss()
    center_criterion = BCEWithLogitsLoss()
    embedding_criterion = MSELoss()
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, (images, masks, centers) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            centers = centers.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            semantic_logits, center_logits, embeddings = model(images)
            
            # Calculate semantic segmentation loss
            semantic_loss = semantic_criterion(semantic_logits, masks)
            
            # Calculate center prediction loss
            center_loss = center_criterion(center_logits, centers)
            
            # Calculate embedding loss (simplified version - pushes embeddings to be 
            # different for different instances and similar for same instance)
            # For single instance per image, we use a simple regularization
            embedding_loss = embedding_criterion(
                embeddings * masks.repeat(1, embeddings.size(1), 1, 1),
                torch.zeros_like(embeddings)
            )
            
            # Combine losses
            loss = semantic_loss + 0.5 * center_loss + 0.1 * embedding_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, masks, centers in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                centers = centers.to(device)
                
                semantic_logits, center_logits, embeddings = model(images)
                
                semantic_loss = semantic_criterion(semantic_logits, masks)
                center_loss = center_criterion(center_logits, centers)
                embedding_loss = embedding_criterion(
                    embeddings * masks.repeat(1, embeddings.size(1), 1, 1),
                    torch.zeros_like(embeddings)
                )
                
                loss = semantic_loss + 0.5 * center_loss + 0.1 * embedding_loss
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, f'instance_unet_checkpoint_epoch_{epoch+1}.pth')
        
        # Visualize predictions every 2 epochs
        if epoch % 2 == 0:
            visualize_instance_predictions(model, val_dataset, device, num_samples=3, epoch=epoch)
    
    # Save final model
    torch.save(model.state_dict(), 'instance_unet_model_final.pth')
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training History")
    plt.savefig("instance_training_history.png")
    plt.show()
    
    return model

def visualize_instance_predictions(model, dataset, device, num_samples=3, epoch=None):
    """Visualize model predictions on random samples from the dataset"""
    model.eval()
    
    plt.figure(figsize=(20, 5 * num_samples))
    
    for i in range(num_samples):
        # Get a random sample
        idx = np.random.randint(0, len(dataset))
        image, true_mask, true_center = dataset[idx]
        
        # Move to device and add batch dimension
        image_tensor = image.unsqueeze(0).to(device)
        
        # Generate prediction
        with torch.no_grad():
            semantic_logits, center_logits, embeddings = model(image_tensor)
            
            # Use very low threshold for semantic segmentation
            semantic_mask = torch.sigmoid(semantic_logits) > 0.002
            center_mask = torch.sigmoid(center_logits) > 0.002
        
        # Move predictions to CPU for visualization
        semantic_mask = semantic_mask.squeeze().cpu().numpy()
        center_mask = center_mask.squeeze().cpu().numpy()
        embeddings = embeddings.squeeze().cpu().numpy()
        
        # Display original image
        plt.subplot(num_samples, 5, i*5 + 1)
        plt.imshow(image.permute(1, 2, 0))
        plt.title("Original Image")
        plt.axis('off')
        
        # Display true mask
        plt.subplot(num_samples, 5, i*5 + 2)
        plt.imshow(true_mask.squeeze(), cmap='gray')
        plt.title("True Mask")
        plt.axis('off')
        
        # Display predicted semantic mask
        plt.subplot(num_samples, 5, i*5 + 3)
        plt.imshow(semantic_mask, cmap='gray')
        plt.title("Semantic Mask")
        plt.axis('off')
        
        # Display predicted center mask
        plt.subplot(num_samples, 5, i*5 + 4)
        plt.imshow(center_mask, cmap='jet')
        plt.title("Center Prediction")
        plt.axis('off')
        
        # Display embedding visualization (use first 3 channels for RGB)
        if embeddings.shape[0] >= 3:
            embedding_vis = np.stack([
                embeddings[0],
                embeddings[1],
                embeddings[2]
            ], axis=-1)
            
            # Normalize to [0, 1] for visualization
            if embedding_vis.max() > embedding_vis.min():
                embedding_vis = (embedding_vis - embedding_vis.min()) / (embedding_vis.max() - embedding_vis.min())
            
            # Zero out background
            embedding_vis = embedding_vis * semantic_mask.reshape(*semantic_mask.shape, 1)
            
            plt.subplot(num_samples, 5, i*5 + 5)
            plt.imshow(embedding_vis)
            plt.title("Embeddings (RGB)")
            plt.axis('off')
        else:
            plt.subplot(num_samples, 5, i*5 + 5)
            plt.text(0.5, 0.5, "Not enough embedding channels for RGB visualization", 
                     ha='center', va='center')
            plt.axis('off')
    
    if epoch is not None:
        plt.suptitle(f"Instance Segmentation After Epoch {epoch+1}")
    else:
        plt.suptitle("Instance Segmentation Predictions")
    
    plt.tight_layout()
    plt.savefig(f"instance_predictions_epoch_{epoch+1 if epoch is not None else 'final'}.png")
    plt.show()

def process_instances(semantic_mask, center_mask, embeddings, threshold=0.3):
    """
    Process model outputs to generate instance segmentation
    
    Args:
        semantic_mask: Binary mask of foreground objects
        center_mask: Predicted centers
        embeddings: Embedding vectors for each pixel
        threshold: Distance threshold for embedding similarity
    
    Returns:
        colored_instances: Image with each instance colored differently
        instance_masks: List of binary masks, one per instance
    """
    # Convert to numpy arrays if tensors
    if isinstance(semantic_mask, torch.Tensor):
        semantic_mask = semantic_mask.cpu().numpy()
    if isinstance(center_mask, torch.Tensor):
        center_mask = center_mask.cpu().numpy()
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    
    # Ensure semantic mask is binary
    semantic_mask = semantic_mask > 0
    
    # Find potential instance centers
    center_mask = center_mask > 0.5
    
    # Use connected components to identify individual centers
    labeled_centers, num_centers = ndimage.label(center_mask)
    
    if num_centers == 0:
        # Fallback to treating the whole semantic mask as one instance
        labeled_centers, num_centers = ndimage.label(semantic_mask)
    
    # Create a colored visualization where each instance has a unique color
    colored_instances = np.zeros((*semantic_mask.shape, 3), dtype=np.uint8)
    instance_masks = []
    
    # Process each detected center
    for i in range(1, num_centers + 1):
        # Get center coordinates
        center_coords = np.where(labeled_centers == i)
        if len(center_coords[0]) == 0:
            continue
            
        # Take the mean position as the center
        cy, cx = int(np.mean(center_coords[0])), int(np.mean(center_coords[1]))
        
        # Skip if center is not on a foreground pixel
        if not semantic_mask[cy, cx]:
            continue
            
        # Get the embedding vector at the center
        center_embedding = embeddings[:, cy, cx]
        
        # Create instance mask
        instance_mask = np.zeros_like(semantic_mask, dtype=bool)
        
        # For each foreground pixel, check if it belongs to this instance
        for y in range(semantic_mask.shape[0]):
            for x in range(semantic_mask.shape[1]):
                if semantic_mask[y, x]:
                    pixel_embedding = embeddings[:, y, x]
                    # Calculate Euclidean distance between embeddings
                    distance = np.linalg.norm(pixel_embedding - center_embedding)
                    if distance < threshold:
                        instance_mask[y, x] = True
        
        # Skip tiny instances
        if np.sum(instance_mask) < 10:
            continue
            
        # Generate a unique color for this instance
        color = np.array([
            (i * 35) % 256,
            (i * 60) % 256,
            (i * 111) % 256
        ], dtype=np.uint8)
        
        # Color the instance in the visualization
        colored_instances[instance_mask] = color
        
        # Add the instance mask to our list
        instance_masks.append(instance_mask)
    
    # If no instances were found, use the semantic mask as fallback
    if not instance_masks:
        color = np.array([255, 0, 0], dtype=np.uint8)
        colored_instances[semantic_mask] = color
        instance_masks.append(semantic_mask)
    
    return colored_instances, instance_masks

if __name__ == "__main__":
    # Train the instance U-Net model
    model = train_instance_unet(epochs=10)
    print("Training completed successfully!")
