import os
from PIL import Image
import math
import multiprocessing
from multiprocessing import Pool, cpu_count
import time
from functools import partial

Image.MAX_IMAGE_PIXELS = None  

INPUT_DIRECTORY = r"e:\Manwa\Grand_Blue\result"
OUTPUT_DIRECTORY = r"e:\Manwa\Grand_Blue\final"

NUM_PROCESSES = 20
BATCH_SIZE = 2            
SEGMENT_HEIGHT = 8000         

# New feature: Split by file size limit
ENABLE_SIZE_BASED_SPLITTING = True   # Enable splitting by file size instead of height
TARGET_FILE_SIZE_KB = 300            # Target file size in KB
MAX_FILE_SIZE_KB = 500               # Maximum allowed file size in KB
MIN_SEGMENT_HEIGHT = 1000            # Minimum height for a segment
SIZE_TOLERANCE = 0.1                 # 10% tolerance for file size (290-330KB acceptable)

FILE_PREFIX = "c"              
FILE_EXTENSION = ".png"        

OUTPUT_FORMAT = "JPEG"         
OUTPUT_EXTENSION = ".jpg"      
JPEG_QUALITY = 95             
JPEG_OPTIMIZE = True          
JPEG_PROGRESSIVE = False      

ENABLE_LARGE_IMAGE_SUPPORT = True    
MAX_MEMORY_USAGE_MB = 2048           

SHOW_DETAILED_LOG = False     # Set to False for less verbose output
SHOW_PROGRESS = True          

def get_image_info(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode
            format_info = img.format
            estimated_memory = width * height * len(mode) if mode else width * height * 3
            estimated_memory_mb = estimated_memory / (1024 * 1024)
            return {
                'width': width,
                'height': height,
                'mode': mode,
                'format': format_info,
                'estimated_memory_mb': estimated_memory_mb,
                'total_pixels': width * height
            }
    except Exception as e:
        return {'error': str(e)}

def estimate_file_size_kb(width, height, format_type="JPEG", quality=95):
    """Estimate file size in KB based on dimensions and format"""
    if format_type == "JPEG":
        # Rough estimation for JPEG: depends on quality and content complexity
        # Base calculation: width * height * compression_factor
        if quality >= 95:
            compression_factor = 0.8
        elif quality >= 85:
            compression_factor = 0.4
        elif quality >= 75:
            compression_factor = 0.2
        else:
            compression_factor = 0.1
        
        estimated_bytes = width * height * compression_factor
        return estimated_bytes / 1024
    else:
        # PNG estimation (much larger)
        estimated_bytes = width * height * 3  # RGB
        return estimated_bytes / 1024

def find_optimal_segment_height(width, target_size_kb, format_type="JPEG", quality=95, min_height=1000):
    """Find the optimal height for a segment to match target file size"""
    # Binary search to find optimal height
    low_height = min_height
    high_height = 15000  # Max reasonable height for a segment
    
    target_tolerance = target_size_kb * SIZE_TOLERANCE
    best_height = low_height
    
    for _ in range(20):  # Max 20 iterations
        mid_height = (low_height + high_height) // 2
        estimated_size = estimate_file_size_kb(width, mid_height, format_type, quality)
        
        if abs(estimated_size - target_size_kb) <= target_tolerance:
            return mid_height
        elif estimated_size < target_size_kb:
            low_height = mid_height + 1
            best_height = mid_height
        else:
            high_height = mid_height - 1
    
    return best_height

def split_image_large(image_info, segment_height, show_detailed_log, output_config):
    image_path, output_dir, image_filename = image_info
    output_format, output_ext, quality, optimize, progressive = output_config
    try:
        img_info = get_image_info(image_path)
        if 'error' in img_info:
            return f"✗ Error reading {image_filename}: {img_info['error']}"
        width = img_info['width']
        height = img_info['height']
        estimated_memory = img_info['estimated_memory_mb']
        total_pixels = img_info['total_pixels']
        
        # Choose splitting method
        if ENABLE_SIZE_BASED_SPLITTING:
            optimal_height = find_optimal_segment_height(width, TARGET_FILE_SIZE_KB, output_format, quality, MIN_SEGMENT_HEIGHT)
            num_segments = math.ceil(height / optimal_height)
            actual_segment_height = optimal_height
            split_method = f"size-based (target: {TARGET_FILE_SIZE_KB}KB, height: {optimal_height}px)"
        else:
            num_segments = math.ceil(height / segment_height)
            actual_segment_height = segment_height
            split_method = f"height-based ({segment_height}px)"
            
        os.makedirs(output_dir, exist_ok=True)
        segment_sizes = []  # Track actual file sizes
        
        with Image.open(image_path) as img:
            if output_format == "JPEG" and img.mode in ("RGBA", "P"):
                for i in range(num_segments):
                    top = int(i * actual_segment_height)
                    bottom = int(min((i + 1) * actual_segment_height, height))
                    width_int = int(width)
                    cropped = img.crop((0, top, width_int, bottom))
                    if cropped.mode in ("RGBA", "P"):
                        background = Image.new("RGB", cropped.size, (255, 255, 255))
                        if cropped.mode == "P":
                            cropped = cropped.convert("RGBA")
                        background.paste(cropped, mask=cropped.split()[-1] if cropped.mode == "RGBA" else None)
                        cropped = background
                    elif cropped.mode != "RGB":
                        cropped = cropped.convert("RGB")
                    
                    output_filename = f"{i + 1}{output_ext}"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Save and check file size
                    cropped.save(output_path, format=output_format, quality=quality, optimize=optimize, progressive=progressive)
                    
                    # Check actual file size
                    actual_size_kb = os.path.getsize(output_path) / 1024
                    segment_sizes.append(actual_size_kb)
                    
                    # If file is too large, try to reduce quality or re-split
                    if ENABLE_SIZE_BASED_SPLITTING and actual_size_kb > MAX_FILE_SIZE_KB:
                        # Try reducing quality
                        reduced_quality = max(60, quality - 20)
                        cropped.save(output_path, format=output_format, quality=reduced_quality, optimize=True, progressive=progressive)
                        actual_size_kb = os.path.getsize(output_path) / 1024
                        segment_sizes[-1] = actual_size_kb
                    
                    del cropped
            else:
                if output_format == "JPEG" and img.mode != "RGB":
                    img = img.convert("RGB")
                for i in range(num_segments):
                    top = int(i * actual_segment_height)
                    bottom = int(min((i + 1) * actual_segment_height, height))
                    width_int = int(width)
                    cropped = img.crop((0, top, width_int, bottom))
                    output_filename = f"{i + 1}{output_ext}"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    if output_format == "JPEG":
                        cropped.save(output_path, format=output_format, quality=quality, optimize=optimize, progressive=progressive)
                    else:
                        cropped.save(output_path, format=output_format, optimize=True)
                    
                    # Check actual file size
                    actual_size_kb = os.path.getsize(output_path) / 1024
                    segment_sizes.append(actual_size_kb)
                    
                    # If file is too large, try to reduce quality
                    if ENABLE_SIZE_BASED_SPLITTING and output_format == "JPEG" and actual_size_kb > MAX_FILE_SIZE_KB:
                        reduced_quality = max(60, quality - 20)
                        cropped.save(output_path, format=output_format, quality=reduced_quality, optimize=True, progressive=progressive)
                        actual_size_kb = os.path.getsize(output_path) / 1024
                        segment_sizes[-1] = actual_size_kb
                    
                    del cropped
        
        # Generate result message with size info
        if ENABLE_SIZE_BASED_SPLITTING:
            avg_size = sum(segment_sizes) / len(segment_sizes) if segment_sizes else 0
            size_range = f"{min(segment_sizes):.1f}-{max(segment_sizes):.1f}KB" if segment_sizes else "N/A"
            return f"✓ Done {image_filename} - {num_segments} segments ({split_method}) [avg: {avg_size:.1f}KB, range: {size_range}]"
        else:
            return f"✓ Done {image_filename} - {num_segments} segments ({output_format}) [{total_pixels:,} pixels]"
    except Exception as e:
        return f"✗ Error processing {image_filename}: {str(e)}"

def process_images_batch(image_batch, segment_height, show_detailed_log, output_config):
    results = []
    for image_info in image_batch:
        result = split_image_large(image_info, segment_height, show_detailed_log, output_config)
        results.append(result)
        import gc
        gc.collect()
    return results

def create_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def process_all_images_multiprocess():
    if not os.path.exists(INPUT_DIRECTORY):
        print(f"❌ Input directory not found: {INPUT_DIRECTORY}")
        return
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    try:
        files = os.listdir(INPUT_DIRECTORY)
    except Exception as e:
        print(f"❌ Error reading directory: {str(e)}")
        return
    image_files = [f for f in files if f.startswith(FILE_PREFIX) and f.endswith(FILE_EXTENSION)]
    if not image_files:
        print(f"❌ No image files found in {INPUT_DIRECTORY}")
        return
    image_info_list = []
    for image_file in sorted(image_files):
        image_path = os.path.join(INPUT_DIRECTORY, image_file)
        folder_name = os.path.splitext(image_file)[0]
        output_dir = os.path.join(OUTPUT_DIRECTORY, folder_name)
        image_info_list.append((image_path, output_dir, image_file))
    output_config = (OUTPUT_FORMAT, OUTPUT_EXTENSION, JPEG_QUALITY, JPEG_OPTIMIZE, JPEG_PROGRESSIVE)
    actual_processes = min(NUM_PROCESSES if NUM_PROCESSES else cpu_count(), len(image_files))
    # Display processing mode
    if ENABLE_SIZE_BASED_SPLITTING:
        print(f"\n🎯 Size-based splitting enabled: Target {TARGET_FILE_SIZE_KB}KB, Max {MAX_FILE_SIZE_KB}KB")
    else:
        print(f"\n📏 Height-based splitting: {SEGMENT_HEIGHT}px per segment")
    
    print(f"🚀 Processing {len(image_files)} images with {actual_processes} processes...")
    batches = list(create_batches(image_info_list, BATCH_SIZE))
    start_time = time.time()
    try:
        with Pool(processes=actual_processes) as pool:
            process_func = partial(
                process_images_batch, 
                segment_height=SEGMENT_HEIGHT,
                show_detailed_log=SHOW_DETAILED_LOG,
                output_config=output_config
            )
            batch_results = pool.map(process_func, batches)
            all_results = []
            for batch_result in batch_results:
                all_results.extend(batch_result)
    except Exception as e:
        print(f"❌ Multiprocessing error: {str(e)}")
        return
    end_time = time.time()
    processing_time = end_time - start_time
    success_count = 0
    error_count = 0
    for result in all_results:
        print(result)
        if result.startswith("✓"):
            success_count += 1
        else:
            error_count += 1
    print(f"\nSummary: Success: {success_count}/{len(image_files)}, Errors: {error_count}/{len(image_files)}")
    print(f"Total time: {processing_time:.2f} seconds, Avg: {processing_time/len(image_files):.2f} sec/image")
    if success_count > 0:
        print(f"Results saved in: {OUTPUT_DIRECTORY}")

if __name__ == "__main__":
    process_all_images_multiprocess() 