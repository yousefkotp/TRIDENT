import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import argparse
from pathlib import Path
import glob

def sample_and_plot_combined_tsne(hard_features, soft_features, min_samples, output_folder, file_name, random_seed):
    """
    Sample features from both hard and soft gridded patches and create combined t-SNE plot.
    
    Args:
        hard_features (np.ndarray): Hard gridded feature array of shape (n_samples, n_features)
        soft_features (np.ndarray): Soft gridded feature array of shape (n_samples, n_features)
        min_samples (int): Minimum number of samples to take
        output_folder (str): Output folder path
        file_name (str): Base filename without extension
        random_seed (int): Random seed for reproducibility
    """
    np.random.seed(random_seed)
    
    # Check if hard features meet minimum requirement
    n_hard_available = hard_features.shape[0]
    n_soft_available = soft_features.shape[0]
    
    if n_hard_available < min_samples:
        print(f"  Warning: Only {n_hard_available} hard samples available (need {min_samples}), skipping")
        return
    
    # Sample from hard features
    n_hard_samples = min(min_samples, n_hard_available)
    hard_indices = np.random.choice(n_hard_available, size=n_hard_samples, replace=False)
    sampled_hard_features = hard_features[hard_indices]
    
    # Sample same number from soft features (or all if less available)
    n_soft_samples = min(n_hard_samples, n_soft_available)
    soft_indices = np.random.choice(n_soft_available, size=n_soft_samples, replace=False)
    sampled_soft_features = soft_features[soft_indices]
    
    print(f"  Sampling {n_hard_samples} hard + {n_soft_samples} soft features (seed: {random_seed})")
    
    # Combine features for t-SNE
    combined_features = np.vstack([sampled_hard_features, sampled_soft_features])
    
    # Create labels for coloring
    labels = np.array(['Hard'] * n_hard_samples + ['Soft'] * n_soft_samples)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=random_seed)
    features_2d = tsne.fit_transform(combined_features)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot hard features
    hard_mask = labels == 'Hard'
    plt.scatter(features_2d[hard_mask, 0], features_2d[hard_mask, 1], 
               alpha=0.6, s=20, c='red', label=f'Hard Gridded ({n_hard_samples})', marker='o')
    
    # Plot soft features
    soft_mask = labels == 'Soft'
    plt.scatter(features_2d[soft_mask, 0], features_2d[soft_mask, 1], 
               alpha=0.6, s=20, c='blue', label=f'Soft Gridded ({n_soft_samples})', marker='^')
    
    plt.title(f't-SNE: Hard vs Soft Gridded Patches - {file_name}\nSample {random_seed}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = os.path.join(output_folder, f'{file_name}_sample_{random_seed}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved plot: {plot_path}")

def process_matching_h5_files(hard_file_path, soft_file_path, min_samples, output_dir):
    """
    Process matching h5 files from hard and soft folders and generate combined t-SNE plots.
    
    Args:
        hard_file_path (str): Path to the hard gridded h5 file
        soft_file_path (str): Path to the soft gridded h5 file
        min_samples (int): Minimum number of samples to take
        output_dir (str): Output directory path
    """
    try:
        print(f"\nProcessing: {Path(hard_file_path).name}")
        
        # Load hard features
        with h5py.File(hard_file_path, 'r') as f:
            if 'features' not in f:
                print(f"  Error: 'features' dataset not found in {hard_file_path}")
                return
            hard_features = f['features'][:]
            print(f"  Hard features shape: {hard_features.shape}")
        
        # Load soft features
        with h5py.File(soft_file_path, 'r') as f:
            if 'features' not in f:
                print(f"  Error: 'features' dataset not found in {soft_file_path}")
                return
            soft_features = f['features'][:]
            print(f"  Soft features shape: {soft_features.shape}")
        
        # Check if features have same dimensionality
        if hard_features.shape[1] != soft_features.shape[1]:
            print(f"  Error: Feature dimensions don't match - Hard: {hard_features.shape[1]}, Soft: {soft_features.shape[1]}")
            return
        
        # Create output folder
        file_name = Path(hard_file_path).stem
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate 5 different samples and plots
        for i in range(5):
            sample_and_plot_combined_tsne(hard_features, soft_features, min_samples, output_dir, file_name, i)
                
    except Exception as e:
        print(f"  Error processing {hard_file_path} and {soft_file_path}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Generate combined t-SNE plots comparing hard vs soft gridded patches from h5 files')
    parser.add_argument('--hard_folder', required=True, help='Folder containing h5 files with hard gridded patch features')
    parser.add_argument('--soft_folder', required=True, help='Folder containing h5 files with soft gridded patch features')
    parser.add_argument('--output_dir', required=True, help='Output directory for plots')
    parser.add_argument('--min_samples', type=int, default=1000, 
                        help='Minimum number of samples required from hard features (default: 1000)')
    
    args = parser.parse_args()
    
    # Validate input folders
    if not os.path.exists(args.hard_folder):
        print(f"Error: Hard folder '{args.hard_folder}' does not exist")
        return
    
    if not os.path.exists(args.soft_folder):
        print(f"Error: Soft folder '{args.soft_folder}' does not exist")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all h5 files in both folders
    hard_pattern = os.path.join(args.hard_folder, '*.h5')
    soft_pattern = os.path.join(args.soft_folder, '*.h5')
    
    hard_files = glob.glob(hard_pattern)
    soft_files = glob.glob(soft_pattern)
    
    # Create dictionaries for easy matching by filename
    hard_files_dict = {Path(f).name: f for f in hard_files}
    soft_files_dict = {Path(f).name: f for f in soft_files}
    
    # Find matching files
    matching_files = set(hard_files_dict.keys()) & set(soft_files_dict.keys())
    
    if not matching_files:
        print("No matching h5 files found between hard and soft folders")
        return
    
    # Sort for consistent processing order
    matching_files = sorted(matching_files)
    
    print(f"Found {len(matching_files)} matching h5 files")
    print(f"Minimum samples required from hard features: {args.min_samples}")
    print(f"Output directory: {args.output_dir}")
    
    # Process each matching file pair
    for file_name in matching_files:
        hard_file_path = hard_files_dict[file_name]
        soft_file_path = soft_files_dict[file_name]
        process_matching_h5_files(hard_file_path, soft_file_path, args.min_samples, args.output_dir)
    
    print(f"\nProcessing complete! Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main() 