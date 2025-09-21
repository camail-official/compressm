#!/usr/bin/env python3
"""
Preprocessing script for PathFinder dataset.
Converts all individual image files into a single NumPy .npz file for fast loading.
"""

import numpy as np
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_pathfinder_dataset(data_dir, output_file=None, resolution=32):
    """
    Preprocess PathFinder dataset by loading all images and storing them in a single .npz file.
    
    Args:
        data_dir (Path): Directory containing the PathFinder dataset
        output_file (Path): Output .npz file path (optional)
        resolution (int): Expected image resolution (32, 64, 128, or 256)
    """
    data_dir = Path(data_dir).expanduser()
    
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")
    
    # Set default output file name
    if output_file is None:
        output_file = data_dir / f"pathfinder{resolution}_preprocessed.npz"
    else:
        output_file = Path(output_file)
    
    logger.info(f"Preprocessing PathFinder dataset from {data_dir}")
    logger.info(f"Output file: {output_file}")
    
    # PathFinder blacklist (same as in original dataset)
    blacklist = {"pathfinder32/curv_baseline/imgs/0/sample_172.png"}
    
    # Collect all image paths and labels
    samples = []
    logger.info("Scanning for image files...")
    
    for diff_level in ["curv_baseline"]:  # Same as original dataset
        metadata_dir = data_dir / diff_level / "metadata"
        if not metadata_dir.exists():
            logger.warning(f"Metadata directory {metadata_dir} does not exist, skipping")
            continue
            
        path_list = sorted(
            list(metadata_dir.glob("*.npy")),
            key=lambda path: int(path.stem),
        )
        
        if not path_list:
            logger.warning(f"No metadata files found in {metadata_dir}")
            continue
            
        for metadata_file in path_list:
            logger.info(f"Processing metadata file: {metadata_file}")
            with open(metadata_file, "r") as f:
                for line in f.read().splitlines():
                    metadata = line.split()
                    if len(metadata) >= 4:
                        image_path = Path(diff_level) / metadata[0] / metadata[1]
                        
                        # Check blacklist
                        if str(Path(data_dir.stem) / image_path) not in blacklist:
                            full_path = data_dir / image_path
                            if full_path.exists():
                                label = int(metadata[3])
                                samples.append((image_path, label))
                            else:
                                logger.warning(f"Image file not found: {full_path}")
    
    logger.info(f"Found {len(samples)} images to process")
    
    if len(samples) == 0:
        raise ValueError("No valid images found in the dataset")
    
    # Load first image to determine dimensions
    first_image_path = data_dir / samples[0][0]
    with Image.open(first_image_path) as img:
        img_gray = img.convert("L")
        img_array = np.array(img_gray)
        height, width = img_array.shape
    
    logger.info(f"Image dimensions: {height}x{width}")
    
    # Prepare arrays to store all data
    logger.info("Loading all images into memory...")
    
    images = np.zeros((len(samples), height, width), dtype=np.uint8)
    labels = np.zeros(len(samples), dtype=np.int32)
    paths = []
    
    # Load and store all images
    for idx, (image_path, label) in enumerate(tqdm(samples, desc="Loading images")):
        full_path = data_dir / image_path
        
        try:
            with Image.open(full_path) as img:
                img_gray = img.convert("L")
                img_array = np.array(img_gray)
                
                images[idx] = img_array
                labels[idx] = label
                paths.append(str(image_path))
                
        except Exception as e:
            logger.error(f"Error loading image {full_path}: {e}")
            raise
    
    # Save to compressed .npz file
    logger.info(f"Saving preprocessed data to {output_file}...")
    np.savez_compressed(
        output_file,
        images=images,
        labels=labels,
        paths=np.array(paths, dtype=object),
        num_samples=len(samples),
        height=height,
        width=width,
        resolution=resolution,
        data_dir=str(data_dir)
    )
    
    logger.info(f"Preprocessing complete! Saved {len(samples)} images to {output_file}")
    
    # Print file size info
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"Output file size: {file_size_mb:.2f} MB")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Preprocess PathFinder dataset")
    parser.add_argument("data_dir", help="PathFinder dataset directory")
    parser.add_argument("-o", "--output", help="Output HDF5 file path")
    parser.add_argument("-r", "--resolution", type=int, default=32, 
                       choices=[32, 64, 128, 256],
                       help="Image resolution (default: 32)")
    
    args = parser.parse_args()
    
    try:
        output_file = preprocess_pathfinder_dataset(
            args.data_dir, 
            args.output, 
            args.resolution
        )
        print(f"Success! Preprocessed dataset saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
