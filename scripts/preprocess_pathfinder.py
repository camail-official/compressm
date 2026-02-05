#!/usr/bin/env python3
"""
Preprocess Pathfinder dataset to create a single .npz file for fast loading.

This script loads all Pathfinder images and saves them into a single .npz file,
which dramatically speeds up training compared to loading individual images from disk.

Usage:
    python scripts/preprocess_pathfinder.py --data-dir /path/to/pathfinder32 --resolution 32
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

def preprocess_pathfinder(data_dir, resolution=32):
    """
    Load all Pathfinder images and save to a single .npz file.
    
    Args:
        data_dir: Path to pathfinder dataset directory
        resolution: Image resolution (32, 64, 128, or 256)
    """
    data_dir = Path(data_dir)
    assert data_dir.exists(), f"Data directory {data_dir} does not exist"
    
    # Blacklist for corrupted files
    blacklist = {f"pathfinder{resolution}/curv_baseline/imgs/0/sample_172.png"}
    
    samples = []
    diff_level = "curv_baseline"
    
    # Find metadata files
    metadata_dir = data_dir / diff_level / "metadata"
    if not metadata_dir.exists():
        print(f"Error: Metadata directory {metadata_dir} not found")
        return False
    
    path_list = sorted(
        list(metadata_dir.glob("*.npy")) + list(metadata_dir.glob("*.txt")),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
    )
    
    if not path_list:
        print(f"Error: No metadata files found in {metadata_dir}")
        return False
    
    print(f"Found {len(path_list)} metadata files")
    print(f"Parsing metadata...")
    
    # Parse metadata to get image paths and labels
    for metadata_file in tqdm(path_list):
        try:
            with open(metadata_file, "r") as f:
                for line in f.read().splitlines():
                    parts = line.split()
                    if len(parts) >= 4:
                        image_rel_path = Path(diff_level) / parts[0] / parts[1]
                        
                        # Check blacklist
                        if str(Path(data_dir.stem) / image_rel_path) not in blacklist:
                            label = int(parts[3])
                            image_full_path = data_dir / image_rel_path
                            samples.append((image_full_path, label))
        except Exception as e:
            print(f"Warning: Failed to parse {metadata_file}: {e}")
    
    print(f"Found {len(samples)} valid samples")
    
    if not samples:
        print("Error: No valid samples found")
        return False
    
    # Load all images
    print(f"Loading {len(samples)} images...")
    images = []
    labels = []
    
    for img_path, label in tqdm(samples):
        try:
            # Load image
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img_array = np.array(img, dtype=np.uint8)
            
            # Verify shape
            if img_array.shape != (resolution, resolution):
                print(f"Warning: Skipping {img_path} - wrong shape {img_array.shape}")
                continue
            
            images.append(img_array)
            labels.append(label)
            
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
    
    # Convert to arrays
    images = np.array(images, dtype=np.uint8)  # (N, H, W)
    labels = np.array(labels, dtype=np.int32)
    
    print(f"Successfully loaded {len(images)} images")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels distribution: {np.bincount(labels)}")
    
    # Save to .npz file
    output_file = data_dir / f"pathfinder{resolution}_preprocessed.npz"
    print(f"Saving preprocessed data to {output_file}...")
    
    np.savez_compressed(
        output_file,
        images=images,
        labels=labels,
        num_samples=len(images),
        resolution=resolution,
    )
    
    print(f"✓ Successfully created {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Pathfinder dataset to create .npz file for fast loading"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to pathfinder dataset directory (e.g., /path/to/pathfinder32)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=32,
        choices=[32, 64, 128, 256],
        help="Image resolution",
    )
    
    args = parser.parse_args()
    
    success = preprocess_pathfinder(args.data_dir, args.resolution)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
