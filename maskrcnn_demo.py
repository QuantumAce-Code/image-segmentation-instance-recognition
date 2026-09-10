import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
import json
import zipfile
import torch
from tqdm import tqdm
import random
import shutil

# Import the MaskRCNNSegmentation class from the main implementation
from mask_rcnn_segmentation import MaskRCNNSegmentation

# The size of the dataset isn't too large
def download_coco_targeted(download_dir='datasets', num_images=6):
    """
    Download a targeted selection of COCO images with specific categories
    """
    print("Setting up targeted COCO dataset...")
    
    # Create directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    # Path to create coco directory
    coco_dir = os.path.join(download_dir, "coco-target")
    os.makedirs(coco_dir, exist_ok=True)
    os.makedirs(os.path.join(coco_dir, "val2017"), exist_ok=True)
    
    # Download annotations if they don't exist
    annotation_file = os.path.join(coco_dir, "annotations", "instances_val2017.json")
    if not os.path.exists(os.path.dirname(annotation_file)):
        os.makedirs(os.path.dirname(annotation_file), exist_ok=True)
        
        # Download just the annotations
        annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        zip_path = os.path.join(download_dir, "annotations.zip")
        
        print("Downloading COCO annotations...")
        try:
            response = requests.get(annotations_url, stream=True)
            if response.status_code == 200:
                with open(zip_path, 'wb') as f:
                    total_size = int(response.headers.get('content-length', 0))
                    for chunk in tqdm(response.iter_content(chunk_size=8192), total=total_size//8192, desc="Downloading annotations"):
                        f.write(chunk)
                
                # Extract annotations
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(coco_dir)
                
                # Remove the zip file
                os.remove(zip_path)
                print("Annotations downloaded and extracted.")
            else:
                print(f"Failed to download annotations: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error downloading annotations: {e}")
            return None
    
    # Use pycocotools to get specific categories
    try:
        from pycocotools.coco import COCO
        
        # Initialize COCO API with the annotation file
        coco = COCO(annotation_file)
        
        # Get images with specific categories like people, cars, dogs, and buses
        # These categories typically produce interesting instance segmentation results
        categories = ['person', 'car', 'dog', 'bus', 'bicycle', 'chair']
        cat_ids = coco.getCatIds(catNms=categories)
        
        print(f"Downloading images for categories: {categories}")
        
        # Get all image IDs for these categories
        img_ids = []
        for cat_id in cat_ids:
            ids = coco.getImgIds(catIds=[cat_id])
            img_ids.extend(ids[:10])  # Get up to 10 images per category
        
        # Remove duplicates and select a subset
        img_ids = list(set(img_ids))
        random.shuffle(img_ids)
        img_ids = img_ids[:num_images]  # Limit to specified number
        
        # Download the selected images
        images_dir = os.path.join(coco_dir, "val2017")
        downloaded_count = 0
        
        for img_id in tqdm(img_ids, desc="Downloading COCO images"):
            # Get image info
            img_info = coco.loadImgs(img_id)[0]
            file_name = img_info['file_name']
            img_path = os.path.join(images_dir, file_name)
            
            # Skip if already downloaded
            if os.path.exists(img_path):
                downloaded_count += 1
                continue
                
            # Download the image
            img_url = f"http://images.cocodataset.org/val2017/{file_name}"
            try:
                response = requests.get(img_url, timeout=10)
                if response.status_code == 200:
                    with open(img_path, 'wb') as f:
                        f.write(response.content)
                    downloaded_count += 1
                    print(f"Downloaded {file_name}")
                else:
                    print(f"Failed to download image {file_name}")
            except Exception as e:
                print(f"Error downloading {file_name}: {e}")
        
        print(f"Successfully downloaded {downloaded_count} images")
        
        # If we couldn't download any images, use a fallback approach
        if downloaded_count == 0:
            print("Falling back to direct image download...")
            return download_coco_direct(download_dir)
        
        return coco_dir
        
    except ImportError:
        print("pycocotools not found, falling back to direct image download")
        return download_coco_direct(download_dir)
    except Exception as e:
        print(f"Error using COCO API: {e}")
        return download_coco_direct(download_dir)

def download_coco_direct(download_dir='datasets'):
    """
    Download a few COCO images directly by URL (fallback method)
    """
    print("Downloading COCO images directly...")
    
    # Create directory structure
    coco_dir = os.path.join(download_dir, "coco-direct")
    images_dir = os.path.join(coco_dir, "val2017")
    os.makedirs(images_dir, exist_ok=True)
    
    # List of reliable COCO image URLs
    image_urls = [
        # Person images
        "http://images.cocodataset.org/val2017/000000581781.jpg",  # Person with bicycle
        "http://images.cocodataset.org/val2017/000000579822.jpg",  # Multiple people
        # Vehicle images
        "http://images.cocodataset.org/val2017/000000397133.jpg",  # Bus
        "http://images.cocodataset.org/val2017/000000037777.jpg",  # Cars
        # Animals
        "http://images.cocodataset.org/val2017/000000105948.jpg",  # Dogs
        "http://images.cocodataset.org/val2017/000000039769.jpg",  # Cat
        # Indoor objects
        "http://images.cocodataset.org/val2017/000000252219.jpg",  # Furniture
        "http://images.cocodataset.org/val2017/000000560974.jpg"   # Kitchen
    ]
    
    # Download each image
    downloaded_count = 0
    for url in tqdm(image_urls, desc="Downloading images"):
        filename = url.split("/")[-1]
        image_path = os.path.join(images_dir, filename)
        
        if not os.path.exists(image_path):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    downloaded_count += 1
                    print(f"Downloaded {filename}")
                else:
                    print(f"Failed to download {filename}: HTTP {response.status_code}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
        else:
            downloaded_count += 1
    
    print(f"Successfully downloaded {downloaded_count} images directly")
    return coco_dir

def download_cityscapes_demo(download_dir='datasets'):
    """
    Download a small sample of Cityscapes dataset for autonomous driving
    """
    print("Downloading Cityscapes demo images...")
    
    # Create directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(os.path.join(download_dir, "cityscapes_demo"), exist_ok=True)
    
    # Sample URLs of public Cityscapes images (these are just examples)
    sample_urls = [
        "https://www.cityscapes-dataset.com/wordpress/wp-content/uploads/2015/07/jena_000029_000019_leftImg8bit.png",
        "https://www.cityscapes-dataset.com/wordpress/wp-content/uploads/2015/07/jena_000014_000019_leftImg8bit.png",
        "https://www.cityscapes-dataset.com/wordpress/wp-content/uploads/2015/07/jena_000000_000019_leftImg8bit.png"
    ]
    
    downloaded_paths = []
    
    for i, url in enumerate(sample_urls):
        file_name = f"cityscapes_sample_{i+1}.png"
        file_path = os.path.join(download_dir, "cityscapes_demo", file_name)
        
        if not os.path.exists(file_path):
            print(f"Downloading sample {i+1}/{len(sample_urls)}...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                downloaded_paths.append(file_path)
            else:
                print(f"Failed to download sample {i+1}")
        else:
            downloaded_paths.append(file_path)
    
    return os.path.join(download_dir, "cityscapes_demo")

def download_pretrained_maskrcnn_model():
    """
    Set up a pre-trained Mask R-CNN model
    """
    print("Setting up pre-trained Mask R-CNN model...")
    
    # For Detectron2, specify the model configuration path in the model zoo format
    # This path will be properly resolved by the model_zoo.get_config_file function
    model_config = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    
    return model_config

def run_maskrcnn_demo(dataset_type="coco", application="general", num_samples=6):
    """
    Run a full demo of Mask R-CNN instance segmentation
    """
    # Step 1: Download appropriate dataset
    if dataset_type == "coco":
        dataset_dir = download_coco_targeted(num_images=num_samples)
        if dataset_dir is None:
            print("Failed to download dataset. Exiting demo.")
            return
    elif dataset_type == "cityscapes":
        dataset_dir = download_cityscapes_demo()
    else:
        print(f"Dataset type {dataset_type} not supported. Using COCO targeted.")
        dataset_dir = download_coco_targeted(num_images=num_samples)
    
    # Step 2: Set up pre-trained model
    model_config = download_pretrained_maskrcnn_model()
    
    # Step 3: Initialize the Mask R-CNN model with Detectron2
    try:
        maskrcnn = MaskRCNNSegmentation(
            num_classes=81,  # COCO has 80 classes + background
            framework="detectron2"
        )
        
        # Load pre-trained weights
        maskrcnn.load_model(model_config)
    except Exception as e:
        print(f"Error initializing Mask R-CNN: {e}")
        return
    
    # Step 4: Get sample images for inference
    if dataset_type == "coco":
        # For COCO, get images from the downloaded set
        images_dir = os.path.join(dataset_dir, "val2017")
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    else:  # Cityscapes
        image_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # If there are too few images, use all of them
    if len(image_files) <= num_samples:
        selected_images = image_files
    else:
        # Otherwise, select a random subset
        selected_images = random.sample(image_files, num_samples)
    
    # Step 5: Run inference and visualize results
    results_dir = "maskrcnn_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Determine the number of rows and columns for the visualization grid
    grid_size = int(np.ceil(np.sqrt(len(selected_images))))
    fig_size = min(15, 5 * grid_size)  # Limit maximum figure size
    
    # Create a figure with subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(fig_size, fig_size))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for i, image_file in enumerate(selected_images):
        if dataset_type == "coco":
            image_path = os.path.join(dataset_dir, "val2017", image_file)
        else:  # Cityscapes
            image_path = os.path.join(dataset_dir, image_file)
        
        # Run prediction
        print(f"Processing image {i+1}/{len(selected_images)}: {image_file}")
        
        try:
            # Get prediction result
            result_image = maskrcnn.predict(image_path)
            
            # Load original image for comparison
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Save the result
            result_file = os.path.join(results_dir, f"result_{i+1}.png")
            cv2.imwrite(result_file, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            
            # Display results in the current subplot
            if i < len(axes):
                # Add both images to the same subplot
                ax = axes[i]
                h, w, _ = original_image.shape
                combined_img = np.zeros((h, w*2, 3), dtype=np.uint8)
                combined_img[:, :w, :] = original_image
                combined_img[:, w:, :] = cv2.resize(result_image, (w, h))
                
                ax.imshow(combined_img)
                ax.set_title(f"Image {i+1}: Original and Segmented")
                ax.axis("off")
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            if i < len(axes):
                axes[i].text(0.5, 0.5, f"Error processing\n{image_file}", 
                            horizontalalignment='center', verticalalignment='center')
                axes[i].axis("off")
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "maskrcnn_demo_results.png"), dpi=150)
    
    # Create a PDF report for presentation
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages(os.path.join(results_dir, "maskrcnn_report.pdf")) as pdf:
            # Create a cover page
            fig = plt.figure(figsize=(8.5, 11))
            plt.text(0.5, 0.5, "Instance Segmentation with Mask R-CNN\nDemo Results", 
                    horizontalalignment='center', verticalalignment='center', fontsize=24)
            plt.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
            
            # Add the overview grid
            pdf.savefig(fig)
            
            # Create individual pages for each result
            for i, image_file in enumerate(selected_images):
                if dataset_type == "coco":
                    image_path = os.path.join(dataset_dir, "val2017", image_file)
                else:
                    image_path = os.path.join(dataset_dir, image_file)
                
                result_file = os.path.join(results_dir, f"result_{i+1}.png")
                
                if os.path.exists(result_file):
                    # Create a new figure for each page
                    fig = plt.figure(figsize=(8.5, 11))
                    
                    # Original image
                    plt.subplot(2, 1, 1)
                    original_image = cv2.imread(image_path)
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                    plt.imshow(original_image)
                    plt.title("Original Image")
                    plt.axis("off")
                    
                    # Segmentation result
                    plt.subplot(2, 1, 2)
                    result_image = cv2.imread(result_file)
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    plt.imshow(result_image)
                    plt.title("Instance Segmentation Result")
                    plt.axis("off")
                    
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
        
        print(f"PDF report saved to {results_dir}/maskrcnn_report.pdf")
    except Exception as e:
        print(f"Error creating PDF report: {e}")
    
    print(f"Mask R-CNN instance segmentation demo completed!")
    print(f"Results saved to {results_dir}/maskrcnn_demo_results.png")

def run_application_specific_demo(application="autonomous_driving"):
    """
    Run application-specific demos for real-world scenarios
    """
    print(f"Running application-specific demo for: {application}")
    
    if application == "autonomous_driving":
        # Download Cityscapes demo images
        dataset_dir = download_cityscapes_demo()
        
        # Initialize Mask R-CNN for autonomous driving
        maskrcnn = MaskRCNNSegmentation(
            num_classes=81,  # COCO has most vehicle/pedestrian classes we need
            framework="detectron2"
        )
        
        # Load pre-trained model
        model_config = download_pretrained_maskrcnn_model()
        maskrcnn.load_model(model_config)
        
        # Run inference on all images
        image_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        results_dir = "autonomous_driving_results"
        os.makedirs(results_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 5 * len(image_files)))
        
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(dataset_dir, image_file)
            
            # Run prediction
            result_image = maskrcnn.predict(image_path)
            
            # Save the result
            result_file = os.path.join(results_dir, f"result_{i}.png")
            cv2.imwrite(result_file, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            
            # Display results
            plt.subplot(len(image_files), 1, i+1)
            plt.imshow(result_image)
            plt.title(f"Autonomous Driving Scene Analysis: {image_file}")
            plt.axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "autonomous_driving_results.png"))
        
        print(f"Autonomous driving demo completed!")
        print(f"Results saved to {results_dir}/autonomous_driving_results.png")

if __name__ == "__main__":
    # Run the Mask R-CNN demo with more samples
    run_maskrcnn_demo(dataset_type="coco", num_samples=6)
    
    # Uncomment to run application-specific demos
    # run_application_specific_demo(application="autonomous_driving")
