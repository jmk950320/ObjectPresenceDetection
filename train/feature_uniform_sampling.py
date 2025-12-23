import os
import sys
import yaml
import torch
import numpy as np
import faiss
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from sklearn.decomposition import PCA

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config



def apply_mask_and_roi(grid_h, grid_w, mask_path, roi, roi_format):
    """
    Apply mask and ROI to the feature map.
    """
    H, W = grid_h, grid_w
    
    # Create a binary mask for valid regions
    valid_mask = np.ones((H, W), dtype=bool)
    
    original_h, original_w = None, None

    # 1. Apply Image Mask if exists
    if mask_path and os.path.exists(mask_path):
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is not None:
            # Apply ROI if exists
            if roi:
                if roi_format == 'xyxy':
                    x1, y1, x2, y2 = roi
                elif roi_format == 'xywh':
                    x1, y1, w_roi, h_roi = roi
                    x2, y2 = x1 + w_roi, y1 + h_roi
                else:
                    print(f"Warning: Unknown roi_format {roi_format}. ROI ignored.")
                    x1, y1, x2, y2 = 0, 0, mask_img.shape[1], mask_img.shape[0]

                # Ensure ROI is within bounds
                h_img, w_img = mask_img.shape[:2]
                x1 = int(max(0, min(w_img, x1)))
                y1 = int(max(0, min(h_img, y1)))
                x2 = int(max(0, min(w_img, x2)))
                y2 = int(max(0, min(h_img, y2)))

                # Crop the mask
                mask_img = mask_img[y1:y2, x1:x2]

            # Resize mask to feature grid size
            if mask_img.size > 0:
                mask_resized = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)
                # Assuming white (255) is valid, black (0) is invalid
                valid_mask = valid_mask & (mask_resized > 127)
            else:
                print("Warning: ROI resulted in empty mask. Setting valid_mask to all False.")
                valid_mask[:] = False

    return valid_mask

def uniform_sampling(features, n_samples):
    """
    Perform uniform sampling using Faiss K-Means.
    features: (N, D) numpy array
    """
    d = features.shape[1]
    kmeans = faiss.Kmeans(d, n_samples, niter=20, verbose=True, gpu=True if torch.cuda.is_available() else False)
    kmeans.train(features)
    
    # Centroids are the sampled representatives
    centroids = kmeans.centroids
    
    # Find nearest actual features to centroids
    index = faiss.IndexFlatL2(d)
    index.add(features)
    D, I = index.search(centroids, 1)
    
    sampled_indices = I.flatten()
    sampled_features = features[sampled_indices]
    
    return sampled_features, sampled_indices

def main():
    parser = argparse.ArgumentParser(description='Faiss Uniform Sampling')
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Config is now flat in sampling_config.yaml
    sampling_cfg = config
    
    feature_dir = sampling_cfg.get('feature_dir')
    mask_path = sampling_cfg.get('mask_path')
    roi = sampling_cfg.get('roi')
    roi_format = sampling_cfg.get('roi_format', 'xyxy')
    grid_h = sampling_cfg.get('grid_h')
    grid_w = sampling_cfg.get('grid_w')
    # n_samples = sampling_cfg.get('n_samples') # Calculated from mask
    output_dir = sampling_cfg.get('output_dir', 'sampling_result')
    pca_dim = sampling_cfg.get('pca_dim', 3) # Default to 2D PCA
    
    if not all([feature_dir, grid_h, grid_w]):
        print("Error: Missing required sampling config (feature_dir, grid_h, grid_w).")
        sys.exit(1)
        
    os.makedirs(output_dir, exist_ok=True)
    
    # ROI is loaded from config
    if roi:
        print(f"Loaded ROI: {roi} (format: {roi_format})")

    # Calculate valid mask and n_samples early
    valid_mask = apply_mask_and_roi(grid_h, grid_w, mask_path, roi, roi_format)
    n_samples = np.sum(valid_mask) * 3
    print(f"Set n_samples to mask ROI size: {n_samples}")

    if n_samples == 0:
        print("Error: No valid pixels in mask ROI.")
        sys.exit(1)
        
    # Process feature files
    feature_files = sorted(list(Path(feature_dir).glob('*.pt')))
    if not feature_files:
        print(f"No .pt files found in {feature_dir}")
        sys.exit(1)
        
    print(f"Found {len(feature_files)} feature files. Aggregating features...")
    
    all_valid_features = []
    
    for i, fpath in enumerate(feature_files):
        try:
            features = torch.load(fpath)
            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy()
            
            # Reshape: (N_patches, Dim) -> (H, W, Dim)
            if features.shape[0] != grid_h * grid_w:
                print(f"Warning: Feature shape {features.shape} in {fpath.name} does not match grid {grid_h}x{grid_w}. Skipping.")
                continue
                
            feature_map = features.reshape(grid_h, grid_w, -1)
            
            # Extract valid features using the mask
            valid_feats = feature_map[valid_mask]
            all_valid_features.append(valid_feats)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(feature_files)} files...")
                
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            continue

    if not all_valid_features:
        print("Error: No valid features extracted.")
        sys.exit(1)
        
    # Concatenate all features
    all_valid_features = np.concatenate(all_valid_features, axis=0)
    print(f"Total aggregated valid features shape: {all_valid_features.shape}")
    
    # Check if we have enough features for sampling
    if all_valid_features.shape[0] < n_samples:
        print(f"Warning: Total features ({all_valid_features.shape[0]}) < n_samples ({n_samples}). Adjusting n_samples.")
        n_samples = all_valid_features.shape[0]

    # Perform Uniform Sampling
    print(f"Performing uniform sampling (n={n_samples})...")
    sampled_feats, sampled_indices = uniform_sampling(all_valid_features, n_samples)
    print(f"Sampled {len(sampled_feats)} features.")
    
    # Perform PCA on Sampled Features
    print(f"Performing PCA (dim={pca_dim}) on sampled features...")
    pca = PCA(n_components=pca_dim, whiten=True)
    pca.fit(sampled_feats)

    
    # Save results
    save_path = os.path.join(output_dir, "aggregated_sampled_pca.npz")
    np.savez(save_path, 
             pca_components=pca.components_,
             pca_mean=pca.mean_,
             pca_explained_variance=pca.explained_variance_ratio_)
             
    print(f"Saved sampled features and PCA results to {save_path}")
    
if __name__ == "__main__":
    main()
