import os
import shutil
import sys
from PIL import Image

Image.MAX_IMAGE_PIXELS = None 
# -----------------------------------------------------
DIR_NAME = "test"
SOURCE_BASE = "h:\manhwa\Rent-A-Girlfriend_1"
TARGET_BASE = "h:\manhwa\Rent-A-Girlfriend_1\chap"
CROP_PIXELS = 0  # Số pixel cần cắt bỏ ở cuối ảnh
# -----------------------------------------------------

def create_directory_structure(source_dir, target_dir):
    """Create the target directory structure with subdirectories."""
    # Create main directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Create subdirectories
    for subdir in ['inpainted', 'result', 'mask']:
        os.makedirs(os.path.join(target_dir, subdir), exist_ok=True)
    
    print(f"Created directory structure in {target_dir}")

def copy_result_files(source_dir, target_dir, crop_pixels=CROP_PIXELS):
    """Copy and crop files from source/result to target/result and target/inpainted."""
    source_result = os.path.join(source_dir, 'result')
    target_result = os.path.join(target_dir, 'result')
    target_inpainted = os.path.join(target_dir, 'inpainted')
    
    # Check if source directory exists
    if not os.path.exists(source_result):
        print(f"Warning: Source directory {source_result} does not exist.")
        return
    
    cropped_count = 0
    total_files = 0
    
    # Copy and crop files from source/result to target/result and target/inpainted
    for file in os.listdir(source_result):
        source_file = os.path.join(source_result, file)
        if os.path.isfile(source_file):
            total_files += 1
            
            # Crop and copy to result folder
            target_result_file = os.path.join(target_result, file)
            if crop_remove_bottom_pixels(source_file, target_result_file, crop_pixels):
                cropped_count += 1
            
            # Crop and copy to inpainted folder
            target_inpainted_file = os.path.join(target_inpainted, file)
            crop_remove_bottom_pixels(source_file, target_inpainted_file, crop_pixels)
    
    print(f"Processed {total_files} result files - cropped and copied to both result and inpainted folders")
    print(f"Successfully removed bottom {crop_pixels}px from {cropped_count} result images")

def crop_remove_bottom_pixels(image_path, output_path, crop_pixels=CROP_PIXELS):
    """Remove bottom pixels from an image (keep the top portion)."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Check if image is tall enough
            if height <= crop_pixels:
                print(f"Warning: Image {os.path.basename(image_path)} is only {height}px tall (≤ {crop_pixels}px), copying original")
                shutil.copy2(image_path, output_path)
                return True
            
            # Remove bottom pixels (keep top portion)
            # Box format: (left, top, right, bottom)
            new_height = height - crop_pixels
            crop_box = (0, 0, width, new_height)
            cropped_img = img.crop(crop_box)
            
            # Save the cropped image
            cropped_img.save(output_path)
            print(f"Cropped {os.path.basename(image_path)}: {width}x{height} → {width}x{new_height} (removed {crop_pixels}px from bottom)")
            return True
            
    except Exception as e:
        print(f"Error cropping image {os.path.basename(image_path)}: {e}")
        return False

def copy_png_files(source_dir, target_dir, crop_pixels=CROP_PIXELS):
    """Copy PNG files from source/vn to target directory with bottom pixels removal."""
    source_vn = os.path.join(source_dir, 'vn')
    
    # Check if source directory exists
    if not os.path.exists(source_vn):
        print(f"Warning: Source directory {source_vn} does not exist.")
        return []
    
    # Copy and crop PNG files
    copied_files = []
    cropped_count = 0
    
    for file in os.listdir(source_vn):
        if file.lower().endswith('.png'):
            source_file = os.path.join(source_vn, file)
            target_file = os.path.join(target_dir, file)
            
            # Remove bottom pixels and save
            if crop_remove_bottom_pixels(source_file, target_file, crop_pixels):
                copied_files.append(file)
                cropped_count += 1
    
    print(f"Processed {len(copied_files)} PNG files from {source_vn} to {target_dir}")
    print(f"Successfully removed bottom {crop_pixels}px from {cropped_count} images")
    return copied_files

def create_mask_files(target_dir, png_files):
    """Create white mask files for each PNG in the target/mask directory."""
    mask_dir = os.path.join(target_dir, 'mask')
    
    for file in png_files:
        source_file = os.path.join(target_dir, file)
        mask_file = os.path.join(mask_dir, file)
        
        try:
            # Open the image to get dimensions
            with Image.open(source_file) as img:
                width, height = img.size
                
                # Create a white image with the same dimensions
                white_img = Image.new('RGB', (width, height), color=(255, 255, 255))
                
                # Save the white image as mask
                white_img.save(mask_file)
        except Exception as e:
            print(f"Error creating mask for {file}: {e}")
    
    print(f"Created {len(png_files)} mask files in {mask_dir}")

def main():
    source_dir = os.path.join(SOURCE_BASE, DIR_NAME)
    target_dir = os.path.join(TARGET_BASE, DIR_NAME)
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} does not exist.")
        return 1
    
    print(f"Processing manga directory: {DIR_NAME}")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    
    # Create target directory structure
    create_directory_structure(source_dir, target_dir)
    
    # Copy result files
    copy_result_files(source_dir, target_dir)
    
    # Copy PNG files and get the list of copied files
    copied_png_files = copy_png_files(source_dir, target_dir)
    
    # Create mask files
    create_mask_files(target_dir, copied_png_files)
    
    print(f"Processing completed successfully for {DIR_NAME}!")
    return 0

if __name__ == "__main__":
    sys.exit(main())