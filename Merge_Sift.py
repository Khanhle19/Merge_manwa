import os
import glob
import shutil
import time
import numpy as np
import cv2
from PIL import Image
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool as ProcessPool
import rasterio
from rasterio.features import shapes, rasterize
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, shape
from shapely import union, intersects, convex_hull

# Disable PIL image size limit
Image.MAX_IMAGE_PIXELS = None

class MangaProcessor:
    def __init__(self, base_path, threads=8):
        """Initialize the manga processor with base path and thread count"""
        self.base_path = base_path
        self.thread_count = threads
        self.target_width = 800
        
        # Set up paths
        self.setup_paths()
        
        # Progress tracking
        self.total_tasks = 0
        self.completed_tasks = 0
        self.start_time = None
        
    def setup_paths(self):
        """Create all necessary directories for the pipeline"""
        print("Setting up directory structure...")
        
        # Main directories
        self.paths = {
            'raw1': os.path.join(self.base_path, 'raw1'),
            'raw2': os.path.join(self.base_path, 'raw2'),
            'raw2_inpainted': os.path.join(self.base_path, 'raw2', 'inpainted'),
            'raw2_mask': os.path.join(self.base_path, 'raw2', 'mask'),
            'raw': os.path.join(self.base_path, 'raw'),
            'raw_mask': os.path.join(self.base_path, 'raw', 'mask'),
            'vn1': os.path.join(self.base_path, 'vn1'),
            'vn2': os.path.join(self.base_path, 'vn2'),
            'vn2_mask': os.path.join(self.base_path, 'vn2', 'mask'),
            'vn3': os.path.join(self.base_path, 'vn3'),
            'vn3_mask': os.path.join(self.base_path, 'vn3', 'mask'),
            'vn': os.path.join(self.base_path, 'vn'),
            'vn_mask': os.path.join(self.base_path, 'vn', 'mask'),
            'mask': os.path.join(self.base_path, 'mask'),
            'result': os.path.join(self.base_path, 'result')
        }
        
        # Create all directories
        for path_name, path in self.paths.items():
            os.makedirs(path, exist_ok=True)
            
        print("✅ Directory structure created")

    @staticmethod
    def natural_sort_key(s):
        import re
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    def start_progress(self, total):
        """Initialize progress tracking"""
        self.total_tasks = total
        self.completed_tasks = 0
        self.start_time = time.time()
    
    def update_progress(self):
        """Update and display progress"""
        self.completed_tasks += 1
        progress = (self.completed_tasks / self.total_tasks) * 100
        elapsed = time.time() - self.start_time
        
        if self.completed_tasks > 0 and elapsed > 0:
            items_per_sec = self.completed_tasks / elapsed
            remaining = (self.total_tasks - self.completed_tasks) / items_per_sec if items_per_sec > 0 else 0
            print(f"Progress: {progress:.1f}% ({self.completed_tasks}/{self.total_tasks}) - ETA: {remaining:.1f}s")
    
    def merge_raw_images(self, dir_to_process=None):
        """Step 1: Merge raw images vertically"""
        print("\n=== Merging Raw Images ===")
        
        # Get directories to process
        dirs = os.listdir(self.paths['raw1'])
        
        # Filter directories if dir_to_process is specified
        if dir_to_process:
            print(f"Processing only directory: {dir_to_process}")
            if dir_to_process in dirs:
                dirs = [dir_to_process]
            else:
                print(f"⚠️ Directory {dir_to_process} not found in raw1")
                return
        
        self.start_progress(len(dirs))
        
        # Define worker function
        def merge_image(dir):
            try:
                name = dir
                padding = 0
                gpath = os.path.join(self.paths['raw2_inpainted'], f"{dir}-*.png")
                image_files = sorted(glob.glob(gpath), key=MangaProcessor.natural_sort_key)
                
                if not image_files:
                    print(f"⚠️ No images found for {dir}")
                    self.update_progress()
                    return
                
                # Calculate total height
                x = 0
                y = 0
                valid_files = []
                
                for path in image_files:
                    try:
                        img = Image.open(path)
                        x = img.size[0]
                        if x > 1000:
                            continue
                        ratio = self.target_width / x
                        y += int(img.size[1] * ratio)
                        valid_files.append(path)
                    except Exception as e:
                        print(f"⚠️ Cannot open image {path}: {e}")
                
                if not valid_files:
                    print(f"⚠️ No valid images found for {dir}")
                    self.update_progress()
                    return
                
                # Create new image
                imgR = Image.new("RGBA", (self.target_width, y))
                y = 0
                
                # Paste each image
                for path in valid_files:
                    try:
                        img = Image.open(path)
                        x = img.size[0]
                        ratio = self.target_width / x
                        y1 = int(img.size[1] * ratio)
                        img = img.resize((self.target_width, y1))
                        
                        if y == 0 and padding > 0:
                            imgR.paste(img.crop((0, padding, self.target_width, y1)), (0, y))
                            y1 = y1 - padding
                        else:
                            imgR.paste(img, (0, y))
                        y += y1
                    except Exception as e:
                        print(f"⚠️ Error processing {path}: {e}")
                
                # Save result
                result_path = os.path.join(self.paths['raw'], f"{name}.png")
                imgR.save(result_path, quality=95)
                print(f"✅ Created merged raw image: {name}.png")
                self.update_progress()
            except Exception as e:
                print(f"❌ Error merging images for {dir}: {e}")
                self.update_progress()
        
        # Process in parallel
        pool = ThreadPool(self.thread_count)
        try:
            pool.map(merge_image, dirs)
        finally:
            pool.close()
            pool.join()
        
        print("Raw image merging completed!")

    def merge_raw_masks(self, dir_to_process=None):
        """Step 2: Merge raw masks vertically"""
        print("\n=== Merging Raw Masks ===")
        
        # Get directories to process
        dirs = os.listdir(self.paths['raw1'])
        
        # Filter directories if dir_to_process is specified
        if dir_to_process:
            print(f"Processing only directory: {dir_to_process}")
            if dir_to_process in dirs:
                dirs = [dir_to_process]
            else:
                print(f"⚠️ Directory {dir_to_process} not found in raw1")
                return
                
        self.start_progress(len(dirs))
        
        # Process each directory
        for dir in dirs:
            try:
                name = dir
                padding = 0
                gpath = os.path.join(self.paths['raw2_mask'], f"{dir}-*.png")
                mask_files = sorted(glob.glob(gpath), key=MangaProcessor.natural_sort_key) 
                
                if not mask_files:
                    print(f"⚠️ No mask files found for {dir}")
                    self.update_progress()
                    continue
                
                # Calculate total height
                x = 0
                y = 0
                valid_files = []
                
                for path in mask_files:
                    try:
                        img = Image.open(path)
                        x = img.size[0]
                        if x > 1000:
                            continue
                        ratio = self.target_width / x
                        y += int(img.size[1] * ratio)
                        valid_files.append(path)
                    except Exception as e:
                        print(f"⚠️ Cannot open mask {path}: {e}")
                
                if not valid_files:
                    print(f"⚠️ No valid mask files found for {dir}")
                    self.update_progress()
                    continue
                
                # Create new mask
                imgR = Image.new("L", (self.target_width, y))
                y = 0
                
                # Paste each mask
                for path in valid_files:
                    try:
                        img = Image.open(path)
                        x = img.size[0]
                        ratio = self.target_width / x
                        y1 = int(img.size[1] * ratio)
                        img = img.resize((self.target_width, y1))
                        
                        if y == 0 and padding > 0:
                            imgR.paste(img.crop((0, padding, self.target_width, y1)), (0, y))
                            y1 = y1 - padding
                        else:
                            imgR.paste(img, (0, y))
                        y += y1
                    except Exception as e:
                        print(f"⚠️ Error processing mask {path}: {e}")
                
                # Save result
                result_path = os.path.join(self.paths['raw_mask'], f"{name}.png")
                imgR.save(result_path)
                print(f"✅ Created merged raw mask: {name}.png")
                self.update_progress()
            except Exception as e:
                print(f"❌ Error merging masks for {dir}: {e}")
                self.update_progress()
        
        print("Raw mask merging completed!")

    def resize_vn_images(self, dir_to_process=None):
        """Step 3: Resize VN images to target width"""
        print("\n=== Resizing VN Images and Masks ===")
        
        # Get files to process
        all_image_files = [f for f in os.listdir(self.paths['vn2']) 
                    if os.path.isfile(os.path.join(self.paths['vn2'], f)) 
                    and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"))]
        
        all_mask_files = []
        if os.path.exists(self.paths['vn2_mask']):
            all_mask_files = [f for f in os.listdir(self.paths['vn2_mask']) 
                         if os.path.isfile(os.path.join(self.paths['vn2_mask'], f)) 
                         and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"))]
        
        # Filter by dir_to_process if specified
        if dir_to_process:
            print(f"Processing only directory: {dir_to_process}")
            
            # Filter image files
            image_files = []
            for f in all_image_files:
                if f.startswith(f"{dir_to_process}-"):
                    image_files.append(f)
            
            # Filter mask files
            mask_files = []
            for f in all_mask_files:
                if f.startswith(f"{dir_to_process}-"):
                    mask_files.append(f)
                    
            if not image_files and not mask_files:
                print(f"⚠️ No files found for {dir_to_process} in vn2 directories")
                return
        else:
            image_files = all_image_files
            mask_files = all_mask_files
        
        total_files = len(image_files) + len(mask_files)
        self.start_progress(total_files)
        
        # Define worker functions
        def resize_image(filename):
            try:
                image_path = os.path.join(self.paths['vn2'], filename)
                output_path = os.path.join(self.paths['vn3'], filename)
                
                if not os.path.isfile(image_path):
                    self.update_progress()
                    return
                    
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    if img.width != self.target_width:
                        ratio = self.target_width / img.width
                        new_height = int(img.height * ratio)
                        img = img.resize((self.target_width, new_height), Image.LANCZOS)
                        img.save(output_path, quality=95)
                    else: 
                        shutil.copy(image_path, output_path)
                print(f"✅ Resized image: {filename}")
                self.update_progress()
            except Exception as e:
                print(f"❌ Error with {filename}: {e}")
                self.update_progress()

        def resize_mask(filename):
            try:
                image_path = os.path.join(self.paths['vn2_mask'], filename)
                output_path = os.path.join(self.paths['vn3_mask'], filename)
                
                if not os.path.isfile(image_path):
                    self.update_progress()
                    return
                    
                with Image.open(image_path) as img:
                    img = img.convert("L")  # Convert to grayscale for masks
                    if img.width != self.target_width:
                        ratio = self.target_width / img.width
                        new_height = int(img.height * ratio)
                        img = img.resize((self.target_width, new_height), Image.LANCZOS)
                        img.save(output_path, quality=95)
                    else: 
                        shutil.copy(image_path, output_path)
                print(f"✅ Resized mask: {filename}")
                self.update_progress()
            except Exception as e:
                print(f"❌ Error with mask {filename}: {e}")
                self.update_progress()
        
        # Process in parallel
        pool = ThreadPool(self.thread_count)
        try:
            if image_files:
                pool.map(resize_image, image_files)
            
            if mask_files:
                pool.map(resize_mask, mask_files)
        finally:
            pool.close()
            pool.join()
        
        print("VN image resizing completed!")

    def merge_vn_images_vertically(self, dir_to_process=None):
        """Step 4: Merge VN images vertically with improved SIFT-based template matching"""
        print("\n=== Merging VN Images Vertically with SIFT ===")
        
        if not os.path.exists(self.paths['vn3']):
            print("⚠️ VN3 directory not found. Skipping vertical merging.")
            return
                
        if dir_to_process:
            print(f"Processing only directory: {dir_to_process}")
            dirs = [dir_to_process]
        else:
            dirs = os.listdir(self.paths['vn1'])
                
        self.start_progress(len(dirs))
        
        def find_template_position_sift(raw_image, template, search_region):
            try:
                y_start, y_end = search_region
                y_start = max(0, y_start)
                y_end = min(raw_image.shape[0], y_end)
                
                if y_start >= y_end or y_start >= raw_image.shape[0] or y_end <= 0:
                    print("⚠️ Invalid search region")
                    return None, None, 0.0
                
                region = raw_image[y_start:y_end, :].copy()
                
                # Resize template if needed
                resize_needed = False
                original_template = template.copy()
                
                if template.shape[1] > region.shape[1]:
                    scale_factor = region.shape[1] / template.shape[1] * 0.95
                    resize_needed = True
                else:
                    scale_factor = 1.0
                    
                if template.shape[0] > region.shape[0]:
                    height_scale = region.shape[0] / template.shape[0] * 0.95
                    if height_scale < scale_factor:
                        scale_factor = height_scale
                    resize_needed = True
                
                if resize_needed:
                    new_width = int(template.shape[1] * scale_factor)
                    new_height = int(template.shape[0] * scale_factor)
                    template = cv2.resize(template, (new_width, new_height))
                    print(f"Template resized to {new_width}x{new_height}")
                
                # Convert to grayscale
                if len(region.shape) == 3:
                    region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                else:
                    region_gray = region
                    
                if len(template.shape) == 3:
                    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                else:
                    template_gray = template
                
                # Initialize SIFT detector
                sift = cv2.SIFT_create()
                
                # Find keypoints and descriptors
                kp1, des1 = sift.detectAndCompute(template_gray, None)
                kp2, des2 = sift.detectAndCompute(region_gray, None)
                
                if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
                    print(f"⚠️ Insufficient SIFT features detected: {len(kp1) if kp1 else 0} in template, {len(kp2) if kp2 else 0} in region")
                    return None, None, 0.0
                
                print(f"Found {len(kp1)} keypoints in template, {len(kp2)} keypoints in region")
                
                # Match descriptors using FLANN
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)
                
                # Apply Lowe's ratio test to find good matches
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                
                print(f"Found {len(good_matches)} good matches out of {len(matches)} total matches")
                
                if len(good_matches) < 10:
                    print("⚠️ Not enough good matches found")
                    return None, None, 0.0
                
                # Extract matched keypoints
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography
                homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if homography is None:
                    print("⚠️ Could not compute homography")
                    return None, None, 0.0
                
                # Calculate confidence
                inliers = np.sum(mask)
                confidence = inliers / len(good_matches)
                
                # Get corners of small image
                h, w = template_gray.shape
                corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                
                # Transform corners to find position in region
                transformed_corners = cv2.perspectiveTransform(corners, homography)
                
                # Adjust corners for region offset
                for i in range(4):
                    transformed_corners[i][0][1] += y_start
                
                # Calculate top-left corner position
                top_left_x = int(transformed_corners[0][0][0])
                top_left_y = int(transformed_corners[0][0][1])
                
                return (top_left_x, top_left_y), transformed_corners, confidence
                    
            except Exception as e:
                print(f"⚠️ SIFT matching error: {e}")
                return None, None, 0.0
        
        def find_template_position_template_matching(raw_image, template, search_region, small_images=None, current_index=None):
            try:
                # Extract the search region
                y_start, y_end = search_region
                y_start = max(0, y_start)
                y_end = min(raw_image.shape[0], y_end)
                region = raw_image[y_start:y_end, :].copy()
                
                # Check if template is too small (height < 1000) and we have additional images
                merged_template = None
                images_used = 0 
                
                if template.shape[0] < 1000 and small_images is not None and current_index is not None:
                    print(f"Template height ({template.shape[0]}) < 1000, attempting to merge with next images")
                    merged_height = template.shape[0]
                    merge_candidates = [template]
                    next_idx = current_index + 1
                    
                    # Find consecutive images to merge until height >= 1000 pixels
                    while merged_height < 1000 and next_idx < len(small_images):
                        next_image = small_images[next_idx]
                        merged_height += next_image.shape[0]
                        merge_candidates.append(next_image)
                        images_used += 1  
                        next_idx += 1
                        
                    if len(merge_candidates) > 1:
                        # Create merged template
                        merged_template = np.zeros((merged_height, template.shape[1], 3), dtype=np.uint8)
                        y_offset = 0
                        
                        for img in merge_candidates:
                            # Check if image is wider than merged template
                            if img.shape[1] > merged_template.shape[1]:
                                scale_factor = merged_template.shape[1] / img.shape[1]
                                new_width = merged_template.shape[1]
                                new_height = int(img.shape[0] * scale_factor)
                                img = cv2.resize(img, (new_width, new_height))
                            
                            # Place image in merged template
                            h = img.shape[0]
                            merged_template[y_offset:y_offset+h, 0:img.shape[1]] = img
                            y_offset += h
                        
                        print(f"Merged {len(merge_candidates)} images into template of size {merged_template.shape[1]}x{merged_template.shape[0]}")
                        template = merged_template
                
                # Resize template if needed
                if template.shape[1] > region.shape[1]:
                    scale_factor = region.shape[1] / template.shape[1] * 0.95
                    new_width = int(template.shape[1] * scale_factor)
                    new_height = int(template.shape[0] * scale_factor)
                    template = cv2.resize(template, (new_width, new_height))
                    
                # Ensure template isn't taller than search region
                if template.shape[0] > region.shape[0]:
                    scale_factor = region.shape[0] / template.shape[0] * 0.95
                    new_width = int(template.shape[1] * scale_factor)
                    new_height = int(template.shape[0] * scale_factor)
                    template = cv2.resize(template, (new_width, new_height))
                
                # Perform template matching
                result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val >= 0.75:  # Only accept if confidence is high enough
                    top_left_x = max_loc[0]
                    top_left_y = max_loc[1] + y_start  # Add offset
                    print(f"Template matching found position at ({top_left_x}, {top_left_y}) with confidence {max_val:.3f}")
                    return (top_left_x, top_left_y), max_val, merged_template, images_used
                else:
                    print(f"Template matching confidence too low: {max_val:.3f}")
                    return None, max_val, merged_template, images_used
                        
            except Exception as e:
                print(f"⚠️ Template matching error: {e}")
                return None, 0.0, None, 0
        
        def merge_images_vertically(dir):
            try:
                name = dir
                folder_path = self.paths['vn3']
                output_name = os.path.join(self.paths['vn'], f"{name}.png")
                output_mask = os.path.join(self.paths['vn_mask'], f"{name}.png")
                raw_path = os.path.join(self.paths['raw'], f"{name}.png")

                if not os.path.exists(raw_path):
                    print(f"⚠️ Raw image not found for {name}")
                    self.update_progress()
                    return
                
                # Find matching files
                files = sorted(
                    [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f"{dir}-" in f],
                    key=MangaProcessor.natural_sort_key
                )
                
                if not files:
                    print(f"⚠️ No images found for {name}")
                    self.update_progress()
                    return
                
                print(f"Found {len(files)} images for chapter {name}")
                
                # Load images and masks
                images = []
                masks = []

                for filename in files:
                    path = os.path.join(folder_path, filename)
                    mask_path = os.path.join(self.paths['vn3_mask'], filename)
                    
                    if not os.path.exists(mask_path):
                        base_name = os.path.splitext(filename)[0]
                        mask_path = os.path.join(self.paths['vn3_mask'], f"{base_name}.png")
                        if not os.path.exists(mask_path):
                            print(f"⚠️ Mask not found for {filename}, creating blank mask")
                            try:
                                img = cv2.imread(path)
                                if img is not None:
                                    blank_mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
                                    masks.append(blank_mask)
                                    images.append(img)
                                else:
                                    print(f"⚠️ Could not read image: {path}")
                            except Exception as e:
                                print(f"⚠️ Error creating blank mask: {e}")
                            continue
                    
                    try:
                        img = cv2.imread(path)
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None and mask is not None:
                            images.append(img)
                            masks.append(mask)
                        else:
                            print(f"⚠️ Error loading {filename} or its mask")
                    except Exception as e:
                        print(f"⚠️ Error opening {filename} or its mask: {e}")

                if not images:
                    print(f"⚠️ No valid image-mask pairs found for {name}")
                    self.update_progress()
                    return

                # Load raw image
                raw_image = cv2.imread(raw_path)
                if raw_image is None:
                    print(f"⚠️ Could not read raw image: {raw_path}")
                    self.update_progress()
                    return
                    
                raw_height, raw_width, _ = raw_image.shape
                print(f"Raw image dimensions: {raw_width}x{raw_height}")

                # Create output canvases
                merged = Image.new("RGB", (raw_width, raw_height), color=(255, 255, 255))
                merged_mask = Image.new("L", (raw_width, raw_height), color=0)

                # Process each image
                current_y = 0
                positions_found = []
                i = 0
                
                while i < len(images):
                    template = images[i].copy()
                    mask_array = masks[i]
                    
                    # Define search region based on current_y
                    if i == 0:
                        search_region = (0, min(5000, raw_height))
                    else:
                        search_region = (max(0, current_y - 10000), min(raw_height, current_y + 10000))
                    
                    print(f"Searching for {files[i]} in region y={search_region[0]} to y={search_region[1]}")
                    
                    # Find position using improved SIFT matching
                    position_result = find_template_position_sift(raw_image, template, search_region)
                    
                    skip_count = 0
                    merged_template = None
                    
                    if position_result and position_result[2] >= 0.8:
                        position, transformed_corners, confidence = position_result
                        method = "SIFT"
                        print(f"✅ SIFT match for {files[i]}: position {position}, confidence: {confidence:.3f}")
                        
                        # Extract corners for precise paste
                        top_left = (int(transformed_corners[0][0][0]), int(transformed_corners[0][0][1]))
                        top_right = (int(transformed_corners[1][0][0]), int(transformed_corners[1][0][1]))
                        bottom_right = (int(transformed_corners[2][0][0]), int(transformed_corners[2][0][1]))
                        bottom_left = (int(transformed_corners[3][0][0]), int(transformed_corners[3][0][1]))
                        
                        # Calculate width and height from transformed corners
                        width = max(abs(top_right[0] - top_left[0]), abs(bottom_right[0] - bottom_left[0]))
                        height = max(abs(bottom_left[1] - top_left[1]), abs(bottom_right[1] - top_right[1]))
                        
                        # Resize template to match transformed size
                        resized_template = cv2.resize(template, (width, height))
                        resized_mask = cv2.resize(mask_array, (width, height))
                        
                    else:
                        _, _, confidence = position_result
                        print(f"Confidence: {confidence:.3f}")
                        # Fallback to template matching
                        fallback_result = find_template_position_template_matching(
                            raw_image, 
                            template, 
                            search_region,
                            small_images=images,
                            current_index=i
                        )
                        
                        if fallback_result and fallback_result[1] >= 0.75:
                            position = fallback_result[0]
                            confidence = fallback_result[1]
                            merged_template = fallback_result[2]
                            method = "Template matching"
                            
                            # If we merged templates, use the merged one
                            if merged_template is not None:
                                print(f"Using merged template for positioning")
                                template = merged_template
                                
                                # Update mask accordingly - create blank mask if necessary
                                merged_mask_array = np.ones((merged_template.shape[0], merged_template.shape[1]), dtype=np.uint8) * 255
                                mask_array = merged_mask_array
                                
                                skip_count = fallback_result[3] if len(fallback_result) > 3 else 0  

                                # Skip next images that were included in merge
                                # template_height = 0
                                # while template_height < merged_template.shape[0] and i + skip_count + 1 < len(images):
                                #     skip_count += 1
                                #     template_height += images[i + skip_count].shape[0]
                                #     if template_height >= merged_template.shape[0]:
                                #         break
                                    
                                print(f"Will skip next {skip_count} images as they were merged")
                            
                            print(f"✅ Template matching for {files[i]}: position {position}, confidence: {confidence:.3f}")
                            
                            # For template matching, use original size
                            top_left = position
                            width, height = template.shape[1], template.shape[0]
                            resized_template = template
                            resized_mask = mask_array
                        else:
                            print(f"⚠️ Could not find position for {files[i]} with sufficient confidence")
                            i += 1
                            continue
                    
                    # Convert to PIL for pasting
                    img_pil = Image.fromarray(cv2.cvtColor(resized_template, cv2.COLOR_BGR2RGB))
                    mask_pil = Image.fromarray(resized_mask)
                    
                    # Paste using top left corner
                    x, y = top_left
                    
                    # Ensure coordinates are within canvas
                    if x < 0 or y < 0 or x + width > raw_width or y + height > raw_height:
                        # Adjust coordinates and size to fit within canvas
                        paste_x = max(0, x)
                        paste_y = max(0, y)
                        paste_width = min(width, raw_width - paste_x)
                        paste_height = min(height, raw_height - paste_y)
                        
                        if paste_width <= 0 or paste_height <= 0:
                            print(f"⚠️ Image would be completely outside canvas, skipping")
                            i += 1
                            continue
                        
                        # Crop image and mask to fit
                        crop_left = paste_x - x if x < 0 else 0
                        crop_top = paste_y - y if y < 0 else 0
                        
                        img_pil = img_pil.crop((crop_left, crop_top, 
                                            crop_left + paste_width, 
                                            crop_top + paste_height))
                        mask_pil = mask_pil.crop((crop_left, crop_top, 
                                                crop_left + paste_width, 
                                                crop_top + paste_height))
                        
                        print(f"Adjusted paste to ({paste_x}, {paste_y}) with size {img_pil.width}x{img_pil.height}")
                        
                        # Paste the cropped image
                        merged.paste(img_pil, (paste_x, paste_y))
                        merged_mask.paste(mask_pil, (paste_x, paste_y))
                    else:
                        # Normal paste - fully within bounds
                        merged.paste(img_pil, (x, y))
                        merged_mask.paste(mask_pil, (x, y))
                    
                    positions_found.append((i, x, y, width, height, method))
                    
                    # Update current_y for next search based on bottom of current image
                    current_y = y + height
                    
                    print(f"✅ Placed image {files[i]} at position ({x}, {y}) using {method}")
                    
                    # Skip images that were merged in template matching
                    if skip_count > 0:
                        i += skip_count + 1
                    else:
                        i += 1
                
                # Save results if any positions were found
                if positions_found:
                    os.makedirs(os.path.dirname(output_name), exist_ok=True)
                    os.makedirs(os.path.dirname(output_mask), exist_ok=True)
                    
                    merged.save(output_name, quality=95)
                    merged_mask.save(output_mask, quality=95)
                    print(f"✅ Created merged VN image and mask: {name}.png with {len(positions_found)} positioned images")
                else:
                    print(f"⚠️ No images could be positioned for {name}")
                
                self.update_progress()
                
            except Exception as e:
                print(f"❌ Error merging VN images for {dir}: {e}")
                import traceback
                traceback.print_exc()
                self.update_progress()
        
        # Process in parallel
        pool = ThreadPool(self.thread_count)
        try:
            pool.map(merge_images_vertically, dirs)
        finally:
            pool.close()
            pool.join()
        
        print("VN image vertical merging with SIFT completed!")

    # Define worker function
    @staticmethod
    def create_mask(path, maskvn_path=None, maksraw_path=None, mask_path=None):
        try:
            name = os.path.basename(path)
            north = 0
            
            # Định nghĩa các hàm GIS nếu chưa được định nghĩa ở nơi khác
            def mask_to_gdf(mask_path, north=0):
                try:
                    image = Image.open(mask_path)
                    mask = np.array(image)
                    transform = rasterio.transform.from_origin(west=0, north=north, xsize=1, ysize=1)
                    
                    polygon_shapes = []
                    for geom, value in shapes(mask, mask=mask, transform=transform):
                        if value == 255:  # White areas in the mask
                            polygon_shapes.append(shape(geom))
                    
                    gdf = gpd.GeoDataFrame(geometry=polygon_shapes)
                    gdf.crs = 3875
                    return gdf
                except Exception as e:
                    print(f"⚠️ Error creating GeoDataFrame: {e}")
                    return None
            
            def groupby_multipoly(df, by, aggfunc="first"):
                try:
                    data = df.drop(labels=df.geometry.name, axis=1)
                    aggregated_data = data.groupby(by=by).agg(aggfunc)
                    
                    def merge_geometries(block):
                        return MultiPolygon(block.values)
                    
                    g = df.groupby(by=by, group_keys=False)[df.geometry.name].agg(merge_geometries)
                    aggregated_geometry = gpd.GeoDataFrame(g, geometry=df.geometry.name, crs=df.crs)
                    aggregated = aggregated_geometry.join(aggregated_data)
                    return aggregated
                except Exception as e:
                    print(f"⚠️ Error grouping polygons: {e}")
                    return None
                    
            def intersects(geoms, other):
                return geoms.intersects(other)
                
            def convex_hull(geoms):
                return geoms.convex_hull
            
            if maskvn_path is None or maksraw_path is None or mask_path is None:
                base_path = os.path.dirname(os.path.dirname(path))
                maskvn_path = os.path.join(base_path, 'vn', 'mask')
                maksraw_path = os.path.join(base_path, 'raw', 'mask')
                mask_path = os.path.join(base_path, 'mask')
            
            # Get GeoDataFrames from masks
            gdf_vn = mask_to_gdf(path, north)
            raw_mask_path = os.path.join(maksraw_path, name)
            
            if not os.path.exists(raw_mask_path):
                print(f"⚠️ Raw mask not found: {raw_mask_path}")
                return
                
            gdf_raw = mask_to_gdf(raw_mask_path)
            
            if gdf_vn is None or gdf_raw is None:
                print(f"⚠️ Failed to create GeoDataFrames for {name}")
                return
            
            # Group raw mask polygons
            gdf_raw['a'] = 1
            grouped = groupby_multipoly(gdf_raw, by='a')
            
            if grouped is None or len(grouped.geometry) == 0:
                print(f"⚠️ Failed to group polygons for {name}")
                return
                
            geo = grouped.geometry.iloc[0]
            
            # Find intersections
            gdf_vn['intersect'] = intersects(gdf_vn.geometry, geo)
            gdf_vn = gdf_vn[gdf_vn['intersect'] == True]
            
            if len(gdf_vn) == 0:
                print(f"⚠️ No intersections found for {name}")
                return
            
            # Create convex hull
            gdf_vn['geo'] = convex_hull(gdf_vn.geometry)
            listshapes = ((geom, 255) for geom in gdf_vn['geo'])
            
            # Rasterize back to image
            transform = rasterio.transform.from_origin(west=0, north=north, xsize=1, ysize=1)
            image = Image.open(path)
            raster = rasterize(shapes=listshapes, out_shape=(image.size[1], image.size[0]), transform=transform, fill=0)
            pil_image = Image.fromarray(np.uint8(raster))
            output_path = os.path.join(mask_path, name)
            pil_image.save(output_path)
            print(f"✅ Created intersection mask: {name}")
        except Exception as e:
            print(f"❌ Error creating mask for {path}: {e}")
            pass
        return path 

    def create_intersection_masks(self, dir_to_process=None, max_height=100000):
        """Step 5: Create intersection masks using GIS with chunking for large images"""
        print("\n=== Creating Intersection Masks ===")
        
        maskvn_path = os.path.join(self.base_path, 'vn', 'mask')
        maksraw_path = os.path.join(self.base_path, 'raw', 'mask')
        mask_path = os.path.join(self.base_path, 'mask')
        temp_path = os.path.join(self.base_path, 'temp_masks')
        
        for path in [maskvn_path, maksraw_path, mask_path, temp_path]:
            os.makedirs(path, exist_ok=True)
        
        # Get mask files to process
        lists = glob.glob(f"{maskvn_path}/*.png")
        
        # Filter by dir_to_process if specified
        if dir_to_process:
            print(f"Processing only directory: {dir_to_process}")
            dir_to_process_file = f"{dir_to_process}.png"
            lists = [path for path in lists if os.path.basename(path) == dir_to_process_file]
            
        if not lists:
            print("⚠️ No mask files found in VN mask directory. Skipping mask intersection.")
            return
            
        self.start_progress(len(lists))
        
        # Define chunked processing function
        def process_mask_in_chunks(vn_mask_path, raw_mask_path, output_mask_path):
            try:
                # Open masks to get dimensions
                vn_mask = Image.open(vn_mask_path)
                raw_mask = Image.open(raw_mask_path)
                
                width, height = vn_mask.size
                print(f"Processing mask of size {width}x{height}")
                
                # If height is reasonable, process normally
                if height <= max_height:
                    # Use existing code
                    self.create_mask(vn_mask_path, maskvn_path, maksraw_path, mask_path)
                    return
                
                # For large images, process in chunks
                print(f"Large mask detected ({height}px height). Processing in chunks...")
                
                # Calculate number of chunks
                num_chunks = (height + max_height - 1) // max_height
                print(f"Will process in {num_chunks} chunks")
                
                # Create final result mask
                final_mask = Image.new("L", (width, height), 0)
                
                # Process each chunk
                for i in range(num_chunks):
                    start_y = i * max_height
                    end_y = min((i + 1) * max_height, height)
                    chunk_height = end_y - start_y
                    
                    print(f"Processing chunk {i+1}/{num_chunks}: rows {start_y}-{end_y}")
                    
                    # Create temporary chunk masks
                    vn_chunk_path = os.path.join(temp_path, f"vn_chunk_{i}.png")
                    raw_chunk_path = os.path.join(temp_path, f"raw_chunk_{i}.png")
                    result_chunk_path = os.path.join(temp_path, f"result_chunk_{i}.png")
                    
                    # Extract and save chunks
                    vn_chunk = vn_mask.crop((0, start_y, width, end_y))
                    vn_chunk.save(vn_chunk_path)
                    
                    raw_chunk = raw_mask.crop((0, start_y, width, end_y))
                    raw_chunk.save(raw_chunk_path)
                    
                    # Process chunk
                    MangaProcessor.create_mask(vn_chunk_path, temp_path, temp_path, temp_path)
                    
                    # Load result and paste into final mask
                    if os.path.exists(result_chunk_path):
                        chunk_result = Image.open(result_chunk_path)
                        final_mask.paste(chunk_result, (0, start_y))
                    else:
                        print(f"⚠️ Failed to process chunk {i+1}")
                
                # Save final mask
                final_mask.save(output_mask_path)
                print(f"✅ Completed chunked mask processing: {os.path.basename(output_mask_path)}")
                
                # Clean up temp files
                for file in os.listdir(temp_path):
                    if file.startswith("vn_chunk_") or file.startswith("raw_chunk_") or file.startswith("result_chunk_"):
                        try:
                            os.remove(os.path.join(temp_path, file))
                        except:
                            pass
            
            except Exception as e:
                print(f"❌ Error in chunked mask processing: {e}")
                import traceback
                traceback.print_exc()
        
        # Process each mask file
        for vn_mask_path in lists:
            name = os.path.basename(vn_mask_path)
            raw_mask_path = os.path.join(maksraw_path, name)
            output_mask_path = os.path.join(mask_path, name)
            
            if not os.path.exists(raw_mask_path):
                print(f"⚠️ Raw mask not found: {raw_mask_path}")
                self.update_progress()
                continue
            
            # Process this mask with chunking
            process_mask_in_chunks(vn_mask_path, raw_mask_path, output_mask_path)
            self.update_progress()
        
        print("Mask intersection completed!")
        
    def generate_final_results(self, dir_to_process=None):
        """Step 6: Generate final results by compositing images"""
        print("\n=== Generating Final Results ===")
        
        # Update paths
        mask_path = os.path.join(self.base_path, 'mask')
        vn_path = os.path.join(self.base_path, 'vn')
        raw_path = os.path.join(self.base_path, 'raw')
        result_path = os.path.join(self.base_path, 'result')
        
        # Find mask files
        lists = glob.glob(f"{mask_path}/*.png")
        
        # Filter by dir_to_process if specified
        if dir_to_process:
            print(f"Processing only directory: {dir_to_process}")
            dir_to_process_file = f"{dir_to_process}.png"
            lists = [path for path in lists if os.path.basename(path) == dir_to_process_file]
            
        if not lists:
            print("⚠️ No mask files found in mask directory. Skipping final results generation.")
            return
            
        self.start_progress(len(lists))
        
        # Define worker function
        def gen_result(path):
            try:
                name = os.path.basename(path)
                source_path = os.path.join(vn_path, name)
                dest_path = os.path.join(raw_path, name)
                output_path = os.path.join(result_path, name)
                
                if not os.path.exists(source_path):
                    print(f"⚠️ Source image not found: {source_path}")
                    self.update_progress()
                    return
                    
                if not os.path.exists(dest_path):
                    print(f"⚠️ Destination image not found: {dest_path}")
                    self.update_progress()
                    return
                
                # Read images
                source_image = cv2.imread(source_path)
                mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                destination_image = cv2.imread(dest_path)
                
                if source_image is None or mask is None or destination_image is None:
                    print(f"⚠️ Failed to read images for {name}")
                    self.update_progress()
                    return
                
                # Ensure mask is binary
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                
                # Adjust destination image size if needed
                h, w = mask.shape
                h1, w1, _ = destination_image.shape
                
                if h1 < h or w1 < w:
                    # Create a blank white image with the mask's dimensions
                    blank_image = np.zeros((h, w, 3), np.uint8)
                    blank_image[:,:] = (255, 255, 255)  # White background
                    
                    # Copy the destination image onto the blank canvas
                    h_to_copy = min(h1, h)
                    w_to_copy = min(w1, w)
                    blank_image[0:h_to_copy, 0:w_to_copy] = destination_image[0:h_to_copy, 0:w_to_copy]
                    destination_image = blank_image
                
                # Ensure source image is at least as large as the mask
                if source_image.shape[0] < h or source_image.shape[1] < w:
                    print(f"⚠️ Source image too small for mask in {name}")
                    self.update_progress()
                    return
                
                # Apply mask to source image
                x_offset = 0  # x-coordinate
                y_offset = 0  # y-coordinate
                
                # Crop the source image using the mask
                cropped_image = cv2.bitwise_and(source_image[:h, :w], source_image[:h, :w], mask=mask)
                
                # Create normalized masks
                cropped_mask = mask / 255.0  # Convert to float and normalize
                background_mask = 1 - cropped_mask  # Invert mask
                
                # Extract region of interest
                region_of_interest = destination_image[y_offset:y_offset+h, x_offset:x_offset+w]
                
                # Apply masks and combine images
                masked_region = region_of_interest * background_mask[:, :, np.newaxis]
                masked_cropped_image = cropped_image * cropped_mask[:, :, np.newaxis]
                combined_image = masked_region + masked_cropped_image
                
                # Place combined image back into destination
                destination_image[y_offset:y_offset+h, x_offset:x_offset+w] = combined_image
                
                # Save result
                cv2.imwrite(output_path, destination_image)
                print(f"✅ Created final result: {name}")
                self.update_progress()
            except Exception as e:
                print(f"❌ Error generating result for {path}: {e}")
                self.update_progress()
                
        # Process in parallel
        pool = ThreadPool(self.thread_count)
        try:
            pool.map(gen_result, lists)
        finally:
            pool.close()
            pool.join()
        
        print("Final results generation completed!")
        
    def run_pipeline(self, dir_to_process=None):
        """Run the complete pipeline"""
        print("\n🚀 Starting Manga Processing Pipeline 🚀")
        print(f"Base path: {self.base_path}")
        print(f"Using {self.thread_count} threads\n")
        
        if dir_to_process:
            print(f"Processing only directory: {dir_to_process}")
        
        start_time = time.time()
        
        # Execute all steps
        # self.merge_raw_images(dir_to_process)
        # self.merge_raw_masks(dir_to_process)
        # self.resize_vn_images(dir_to_process)
        # self.merge_vn_images_vertically(dir_to_process)
        self.create_intersection_masks(dir_to_process)
        self.generate_final_results(dir_to_process)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Pipeline completed in {elapsed:.1f} seconds!")
        print(f"Final results are available in: {self.paths['result']}")

# Main execution
if __name__ == "__main__":
    base_path = r"h:\manhwa\The_Martial_God_Who_Regressed_Back_to_Level_2"
    
    processor = MangaProcessor(base_path, threads=12)
    
    dir_to_process = "c1"
    
    # Chạy pipeline với thư mục đã chỉ định
    processor.run_pipeline(dir_to_process)
    
    # processor.run_pipeline()