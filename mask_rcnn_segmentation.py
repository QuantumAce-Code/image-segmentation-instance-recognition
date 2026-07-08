import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import time
import random
from PIL import Image
import warnings
import sys

# Add path to Detectron2 installation
sys.path.append(r"C:\Users\acesi\Desktop\Projects\detectron2")

# Suppress warnings for demo purposes
warnings.filterwarnings("ignore")

class MaskRCNNSegmentation:
    def __init__(self, num_classes, device=None, framework="detectron2"):
        """
        Initialize MaskRCNN segmentation model
        
        Args:
            num_classes: Number of classes including background
            device: Device to use (if None, will auto-detect)
            framework: Only "detectron2" is supported
        """
        self.num_classes = num_classes
        
        if framework != "detectron2":
            raise ValueError("Only 'detectron2' framework is supported for production-quality results")
        
        self.framework = framework
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize Detectron2 model
        self.init_detectron2()
    
    def init_detectron2(self):
        """Initialize Detectron2 MaskRCNN model"""
        try:
            from detectron2.config import get_cfg
            from detectron2 import model_zoo
            from detectron2.engine import DefaultPredictor
            
            # Create config
            self.cfg = get_cfg()
            
            # Load base config from model zoo using the get_config_file function
            config_path = model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            self.cfg.merge_from_file(config_path)
            
            # Set number of classes
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes - 1  # Detectron2 doesn't count background class
            
            # Set score threshold for detection
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
            
            # Set model weights from COCO pretrained model
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            
            # Default configs for training
            self.cfg.SOLVER.IMS_PER_BATCH = 2
            self.cfg.SOLVER.BASE_LR = 0.00025
            self.cfg.SOLVER.MAX_ITER = 1000
            self.cfg.SOLVER.STEPS = []
            
            # Initialize predictor
            self.predictor = DefaultPredictor(self.cfg)
            print("Successfully initialized Detectron2 with Mask R-CNN")
            
        except ImportError as e:
            print(f"ERROR: Could not import Detectron2. Make sure it's installed correctly: {e}")
            print("\nPlease install Detectron2 with the following commands:")
            print("pip install torch torchvision")
            print("pip install cython pycocotools")
            print("cd C:\\Users\\acesi\\Desktop\\Projects")
            print("pip install -e detectron2")
            raise ImportError("Detectron2 is required - PyTorch fallback has been disabled")
    
    def register_dataset(self, dataset_name, dataset_dict_fn, metadata=None):
        """Register a dataset with Detectron2"""
        try:
            from detectron2.data import DatasetCatalog, MetadataCatalog
            
            DatasetCatalog.register(dataset_name, dataset_dict_fn)
            if metadata:
                MetadataCatalog.get(dataset_name).set(**metadata)
        except ImportError:
            raise ImportError("Detectron2 not found. Cannot register dataset.")
    
    def train_detectron2(self, train_dataset_name, val_dataset_name=None, output_dir="./output"):
        """Train Detectron2 Mask R-CNN model"""
        try:
            from detectron2.engine import DefaultTrainer
            
            # Set output directory
            self.cfg.OUTPUT_DIR = output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Set training dataset
            self.cfg.DATASETS.TRAIN = (train_dataset_name,)
            
            # Set validation dataset if provided
            if val_dataset_name:
                self.cfg.DATASETS.TEST = (val_dataset_name,)
            else:
                self.cfg.DATASETS.TEST = ()
            
            # Create trainer and start training
            trainer = DefaultTrainer(self.cfg)
            trainer.resume_or_load(resume=False)
            trainer.train()
            
            # Load the best model and initialize predictor
            self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
            self._create_predictor()
            
            return trainer
        except ImportError:
            raise ImportError("Detectron2 not found. Cannot train model.")
    
    def _create_predictor(self):
        """Create a Detectron2 predictor from current config"""
        try:
            from detectron2.engine import DefaultPredictor
            self.predictor = DefaultPredictor(self.cfg)
        except ImportError:
            raise ImportError("Detectron2 not found. Cannot create predictor.")
    
    def evaluate_detectron2(self, dataset_name):
        """Evaluate Detectron2 Mask R-CNN model"""
        try:
            from detectron2.evaluation import COCOEvaluator, inference_on_dataset
            from detectron2.data import build_detection_test_loader
            
            # Create evaluator
            evaluator = COCOEvaluator(dataset_name, self.cfg, False, output_dir=self.cfg.OUTPUT_DIR)
            
            # Create test loader
            test_loader = build_detection_test_loader(self.cfg, dataset_name)
            
            # Run evaluation
            results = inference_on_dataset(self.predictor.model, test_loader, evaluator)
            
            return results
        except ImportError:
            raise ImportError("Detectron2 not found. Cannot evaluate model.")
    
    def save_model(self, path):
        """Save model weights"""
        try:
            # For Detectron2, we save the config
            with open(path + "_config.yaml", "w") as f:
                f.write(self.cfg.dump())
            print(f"Detectron2 config saved to {path}_config.yaml")
            print(f"Model weights are at: {self.cfg.MODEL.WEIGHTS}")
        except Exception as e:
            print(f"Error saving Detectron2 model: {e}")
    
    def load_model(self, path):
        """Load model weights"""
        try:
            from detectron2 import model_zoo
            
            if path.endswith(".yaml"):
                # Always use model_zoo to get the full config path
                config_path = model_zoo.get_config_file(path)
                self.cfg.merge_from_file(config_path)
                
                # Also set the weights to the appropriate URL from model zoo
                self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(path)
            else:
                # If it's a direct weights path
                self.cfg.MODEL.WEIGHTS = path
            
            self._create_predictor()
            print(f"Loaded Detectron2 model from {path}")
        except Exception as e:
            print(f"Error loading Detectron2 model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
    
    def predict(self, image_path, threshold=0.5):
        """
        Perform instance segmentation prediction with clear visualization
        
        Args:
            image_path: Path to the image file
            threshold: Confidence threshold for displaying predictions
            
        Returns:
            Original image with predictions overlaid
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run Detectron2 prediction
            outputs = self.predictor(image)
            
            # Get Detectron2 metadata
            from detectron2.utils.visualizer import Visualizer, ColorMode
            from detectron2.data import MetadataCatalog
            
            try:
                # Use COCO metadata for consistent class names
                metadata = MetadataCatalog.get("coco_2017_val")
            except:
                # If COCO metadata not available, create basic metadata
                from detectron2.data import Metadata
                metadata = Metadata()
                metadata.set(thing_classes=["object"] + [f"class_{i}" for i in range(1, self.num_classes)])
            
            # Create enhanced visualization with prominent masks
            v = Visualizer(
                image,
                metadata,
                scale=1.2,  # Slightly enlarge the visualization
                instance_mode=ColorMode.SEGMENTATION  # Use distinct colors for masks
            )
            instances = outputs["instances"].to("cpu")
            result = v.draw_instance_predictions(instances)
            result_image = result.get_image()
            
            return result_image
                
        except Exception as e:
            print(f"Error in prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}")
    
    def optimize_for_inference(self, export_format="onnx", output_path="optimized_model"):
        """
        Optimize model for inference
        
        Args:
            export_format: Format to export ("onnx")
            output_path: Path to save the optimized model
        """
        # For Detectron2, export options are more limited
        if export_format == "onnx":
            try:
                from detectron2.export.torchscript import export_torchscript_with_instances
                
                # Create dummy input
                sample_input = torch.rand(1, 3, 640, 640).to(self.device)
                
                # Export model to TorchScript first
                scripted_model = export_torchscript_with_instances(self.cfg, self.predictor.model, sample_input)
                
                # Export to ONNX
                torch.onnx.export(
                    scripted_model,
                    sample_input,
                    f"{output_path}.onnx",
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                )
                print(f"Exported Detectron2 model to ONNX at {output_path}.onnx")
            except Exception as e:
                print(f"Error exporting Detectron2 model to ONNX: {e}")
        
        else:
            print(f"Export format {export_format} not supported for Detectron2")
    
    def benchmark_inference(self, image_path, num_iterations=100):
        """
        Benchmark inference speed
        
        Args:
            image_path: Path to a sample image
            num_iterations: Number of inference iterations for benchmarking
            
        Returns:
            Average inference time in milliseconds
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Warm-up
            _ = self.predictor(image)
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                _ = self.predictor(image)
            end_time = time.time()
            
            # Calculate average time
            avg_time = (end_time - start_time) * 1000 / num_iterations  # in milliseconds
            
            print(f"Average inference time: {avg_time:.2f} ms per image")
            print(f"FPS: {1000 / avg_time:.2f}")
            
            return avg_time
            
        except Exception as e:
            print(f"Error in benchmarking: {e}")
            return 0



