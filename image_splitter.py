import os
from PIL import Image
import math
import multiprocessing
from multiprocessing import Pool, cpu_count
import time
from functools import partial

Image.MAX_IMAGE_PIXELS = None  

INPUT_DIRECTORY = r"h:\manhwa\The_Martial_God_Who_Regressed_Back_to_Level_2\result"
OUTPUT_DIRECTORY = r"h:\manhwa\The_Martial_God_Who_Regressed_Back_to_Level_2\finish"

NUM_PROCESSES = 20
BATCH_SIZE = 2            
SEGMENT_HEIGHT = 8000         

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
        num_segments = math.ceil(height / segment_height)
        os.makedirs(output_dir, exist_ok=True)
        with Image.open(image_path) as img:
            if output_format == "JPEG" and img.mode in ("RGBA", "P"):
                for i in range(num_segments):
                    top = int(i * segment_height)
                    bottom = int(min((i + 1) * segment_height, height))
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
                    cropped.save(output_path, format=output_format, quality=quality, optimize=optimize, progressive=progressive)
                    del cropped
            else:
                if output_format == "JPEG" and img.mode != "RGB":
                    img = img.convert("RGB")
                for i in range(num_segments):
                    top = int(i * segment_height)
                    bottom = int(min((i + 1) * segment_height, height))
                    width_int = int(width)
                    cropped = img.crop((0, top, width_int, bottom))
                    output_filename = f"{i + 1}{output_ext}"
                    output_path = os.path.join(output_dir, output_filename)
                    if output_format == "JPEG":
                        cropped.save(output_path, format=output_format, quality=quality, optimize=optimize, progressive=progressive)
                    else:
                        cropped.save(output_path, format=output_format, optimize=True)
                    del cropped
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
    print(f"\n🚀 Processing {len(image_files)} images with {actual_processes} processes...")
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