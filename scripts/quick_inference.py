#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quick_inference.py

Quick inference script for CG-HCAN model.
Provides a simple interface for running inference on single images or directories.
"""

import os
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import DualEncoderUNetPP
from dataProcess import normalize_image

def parse_args():
    parser = argparse.ArgumentParser(description='Quick inference with CG-HCAN')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                      help='Input image path or directory')
    parser.add_argument('--output', type=str, default='./quick_results',
                      help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0',
                      help='Device to use for inference')
    parser.add_argument('--use_tta', action='store_true',
                      help='Use test-time augmentation')
    parser.add_argument('--save_vis', action='store_true',
                      help='Save visualization images')
    parser.add_argument('--model_version', type=str, default='detailed',
                      choices=['simple', 'detailed'],
                      help='Model version for CLIP embeddings')
    
    return parser.parse_args()

def load_model(model_path, device):
    """Load trained CG-HCAN model."""
    print(f"Loading model from {model_path}")
    
    # Initialize model
    model = DualEncoderUNetPP(
        encoder_name='efficientnet-b0',
        encoder_weights='imagenet',
        in_channels=3,
        classes_level2=6,
        classes_level3=16,
        use_bicaf_fusion=True,
        bicaf_reduction=8,
        use_clip_gnn=True
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model

def preprocess_image(image_path):
    """Preprocess input image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size (modify as needed)
    image = cv2.resize(image, (512, 512))
    
    # Normalize
    image = normalize_image(image)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    
    return image_tensor

def predict_single_image(model, image_tensor, primary_mask, device, use_tta=False):
    """Predict on a single image."""
    image_tensor = image_tensor.unsqueeze(0).to(device)
    primary_mask = primary_mask.unsqueeze(0).to(device)
    
    with torch.no_grad():
        if use_tta:
            # Test-time augmentation
            predictions_level2 = []
            predictions_level3 = []
            
            # Original
            pred_l2, pred_l3 = model(image_tensor, primary_mask)
            predictions_level2.append(pred_l2)
            predictions_level3.append(pred_l3)
            
            # Horizontal flip
            img_flip = torch.flip(image_tensor, dims=[3])
            mask_flip = torch.flip(primary_mask, dims=[3])
            pred_l2, pred_l3 = model(img_flip, mask_flip)
            pred_l2 = torch.flip(pred_l2, dims=[3])
            pred_l3 = torch.flip(pred_l3, dims=[3])
            predictions_level2.append(pred_l2)
            predictions_level3.append(pred_l3)
            
            # Average predictions
            final_pred_l2 = torch.mean(torch.stack(predictions_level2), dim=0)
            final_pred_l3 = torch.mean(torch.stack(predictions_level3), dim=0)
        else:
            final_pred_l2, final_pred_l3 = model(image_tensor, primary_mask)
    
    # Convert to numpy
    pred_l2_np = torch.softmax(final_pred_l2, dim=1).cpu().numpy()[0]
    pred_l3_np = torch.softmax(final_pred_l3, dim=1).cpu().numpy()[0]
    
    # Get class predictions
    pred_l2_class = np.argmax(pred_l2_np, axis=0)
    pred_l3_class = np.argmax(pred_l3_np, axis=0)
    
    return pred_l2_class, pred_l3_class, pred_l2_np, pred_l3_np

def create_visualization(original_image, pred_l2, pred_l3, output_path):
    """Create and save visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Level-2 prediction
    im2 = axes[1].imshow(pred_l2, cmap='viridis')
    axes[1].set_title('Level-2 Prediction')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    # Level-3 prediction
    im3 = axes[2].imshow(pred_l3, cmap='plasma')
    axes[2].set_title('Level-3 Prediction')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, device)
    
    # Get input files
    input_path = Path(args.input)
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = list(input_path.glob('*.tif')) + list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
    else:
        raise ValueError(f"Invalid input path: {input_path}")
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"Processing ({i+1}/{len(image_files)}): {image_path.name}")
        
        try:
            # Preprocess image
            image_tensor = preprocess_image(image_path)
            
            # Create dummy primary mask (you should load actual mask if available)
            primary_mask = torch.ones(1, 512, 512)  # Dummy mask
            
            # Predict
            pred_l2, pred_l3, prob_l2, prob_l3 = predict_single_image(
                model, image_tensor, primary_mask, device, args.use_tta
            )
            
            # Save predictions
            output_name = image_path.stem
            
            # Save prediction arrays
            np.save(output_dir / f"{output_name}_level2.npy", pred_l2)
            np.save(output_dir / f"{output_name}_level3.npy", pred_l3)
            
            # Save probability maps
            np.save(output_dir / f"{output_name}_prob_level2.npy", prob_l2)
            np.save(output_dir / f"{output_name}_prob_level3.npy", prob_l3)
            
            # Create visualization if requested
            if args.save_vis:
                # Load original image for visualization
                orig_img = cv2.imread(str(image_path))
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                orig_img = cv2.resize(orig_img, (512, 512))
                
                vis_path = output_dir / f"{output_name}_visualization.png"
                create_visualization(orig_img, pred_l2, pred_l3, vis_path)
            
            print(f"  -> Results saved to {output_dir}")
            
        except Exception as e:
            print(f"  -> ERROR: {e}")
            continue
    
    print(f"\nProcessing completed! Results saved in: {output_dir}")

if __name__ == '__main__':
    main()
