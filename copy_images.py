import os
import shutil
import sys
from PIL import Image
from multiprocessing import Pool, cpu_count

# Try to import tqdm for progress bars, fallback if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Create a simple fallback class
    class tqdm:
        def __init__(self, total=None, desc="", unit=""):
            self.total = total
            self.desc = desc
            self.count = 0
            
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            pass
            
        def update(self, n=1):
            self.count += n
            
        @staticmethod
        def write(text):
            pass

Image.MAX_IMAGE_PIXELS = None 
# -----------------------------------------------------
DIR_NAME = "Grand_Blue"
SOURCE_BASE = r"e:\Manwa"
TARGET_BASE = r"e:\Manwa_result"
CROP_PIXELS = 0
NUM_PROCESSES = min(8, cpu_count())
# -----------------------------------------------------

def create_directory_structure(source_dir, target_dir):
    """Create the target directory structure with subdirectories."""
    # Create main directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Create subdirectories
    for subdir in ['inpainted', 'result', 'mask']:
        os.makedirs(os.path.join(target_dir, subdir), exist_ok=True)

def process_result_file(args):
    """Worker function for processing a single result file."""
    source_file, target_result_file, target_inpainted_file, crop_pixels = args
    
    try:
        # Crop and copy to result folder
        result_success = crop_remove_bottom_pixels(source_file, target_result_file, crop_pixels, verbose=False)
        
        # Crop and copy to inpainted folder
        inpainted_success = crop_remove_bottom_pixels(source_file, target_inpainted_file, crop_pixels, verbose=False)
        
        return result_success and inpainted_success
    except Exception as e:
        tqdm.write(f"❌ Error processing {os.path.basename(source_file)}: {e}")
        return False

def copy_result_files(source_dir, target_dir, crop_pixels=CROP_PIXELS):
    """Copy and crop files from source/result to target/result and target/inpainted using multiprocessing."""
    source_result = os.path.join(source_dir, 'result')
    target_result = os.path.join(target_dir, 'result')
    target_inpainted = os.path.join(target_dir, 'inpainted')
    
    # Check if source directory exists
    if not os.path.exists(source_result):
        return
    
    # Prepare arguments for multiprocessing
    file_args = []
    for file in os.listdir(source_result):
        source_file = os.path.join(source_result, file)
        if os.path.isfile(source_file):
            target_result_file = os.path.join(target_result, file)
            target_inpainted_file = os.path.join(target_inpainted, file)
            file_args.append((source_file, target_result_file, target_inpainted_file, crop_pixels))
    
    if not file_args:
        return
    
    # Process files with multiprocessing and progress bar
    with Pool(NUM_PROCESSES) as pool:
        with tqdm(total=len(file_args), desc="Processing result files", unit="file") as pbar:
            results = []
            for result in pool.imap(process_result_file, file_args):
                results.append(result)
                pbar.update(1)
    
    successful_count = sum(results)

def crop_remove_bottom_pixels(image_path, output_path, crop_pixels=CROP_PIXELS, verbose=True):
    """Remove bottom pixels from an image (keep the top portion)."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Check if image is tall enough
            if height <= crop_pixels:
                if verbose:
                    tqdm.write(f"⚠️  Image {os.path.basename(image_path)} is only {height}px tall (≤ {crop_pixels}px), copying original")
                shutil.copy2(image_path, output_path)
                return True
            
            # Remove bottom pixels (keep top portion)
            # Box format: (left, top, right, bottom)
            new_height = height - crop_pixels
            crop_box = (0, 0, width, new_height)
            cropped_img = img.crop(crop_box)
            
            # Save the cropped image
            cropped_img.save(output_path)
            return True
            
    except Exception as e:
        if verbose:
            tqdm.write(f"❌ Error cropping image {os.path.basename(image_path)}: {e}")
        return False

def process_png_file(args):
    """Worker function for processing a single PNG file."""
    source_file, target_file, crop_pixels = args
    
    try:
        if crop_remove_bottom_pixels(source_file, target_file, crop_pixels, verbose=False):
            return os.path.basename(source_file)
        return None
    except Exception as e:
        tqdm.write(f"❌ Error processing {os.path.basename(source_file)}: {e}")
        return None

def copy_png_files(source_dir, target_dir, crop_pixels=CROP_PIXELS):
    """Copy PNG files from source/vn to target directory with bottom pixels removal using multiprocessing."""
    source_vn = os.path.join(source_dir, 'vn')
    
    # Check if source directory exists
    if not os.path.exists(source_vn):
        return []
    
    # Prepare arguments for multiprocessing
    file_args = []
    for file in os.listdir(source_vn):
        if file.lower().endswith('.png'):
            source_file = os.path.join(source_vn, file)
            target_file = os.path.join(target_dir, file)
            file_args.append((source_file, target_file, crop_pixels))
    
    if not file_args:
        return []
    
    # Process files with multiprocessing and progress bar
    with Pool(NUM_PROCESSES) as pool:
        with tqdm(total=len(file_args), desc="Processing PNG files", unit="file") as pbar:
            results = []
            for result in pool.imap(process_png_file, file_args):
                results.append(result)
                pbar.update(1)
    
    # Filter successful results
    copied_files = [filename for filename in results if filename is not None]
    return copied_files

def create_single_mask(args):
    """Worker function for creating a single mask file."""
    source_file, mask_file = args
    
    try:
        # Open the image to get dimensions
        with Image.open(source_file) as img:
            width, height = img.size
            
            # Create a white image with the same dimensions
            white_img = Image.new('RGB', (width, height), color=(255, 255, 255))
            
            # Save the white image as mask
            white_img.save(mask_file)
            return True
    except Exception as e:
        tqdm.write(f"❌ Error creating mask for {os.path.basename(source_file)}: {e}")
        return False

def create_mask_files(target_dir, png_files):
    """Create white mask files for each PNG in the target/mask directory using multiprocessing."""
    mask_dir = os.path.join(target_dir, 'mask')
    
    # Prepare arguments for multiprocessing
    mask_args = []
    for file in png_files:
        source_file = os.path.join(target_dir, file)
        mask_file = os.path.join(mask_dir, file)
        mask_args.append((source_file, mask_file))
    
    if not mask_args:
        return
    
    # Create masks with multiprocessing and progress bar
    with Pool(NUM_PROCESSES) as pool:
        with tqdm(total=len(mask_args), desc="Creating mask files", unit="file") as pbar:
            results = []
            for result in pool.imap(create_single_mask, mask_args):
                results.append(result)
                pbar.update(1)
    
    successful_count = sum(results)

def main():
    source_dir = os.path.join(SOURCE_BASE, DIR_NAME)
    target_dir = os.path.join(TARGET_BASE, DIR_NAME)
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        return 1
    
    # Create target directory structure
    create_directory_structure(source_dir, target_dir)
    
    # Copy result files
    copy_result_files(source_dir, target_dir)
    
    # Copy PNG files and get the list of copied files
    copied_png_files = copy_png_files(source_dir, target_dir)
    
    # Create mask files
    create_mask_files(target_dir, copied_png_files)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())