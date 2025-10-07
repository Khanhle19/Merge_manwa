import os
import glob
import shutil
import time
import numpy as np
import cv2
from PIL import Image
from multiprocessing.dummy import Pool as ThreadPool

# Disable PIL image size limit
Image.MAX_IMAGE_PIXELS = None

def closure_mask(mask, kernel_size=100, dilation_size=50):
    """Apply morphological closure and dilation to mask"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if dilation_size > 0:
        dilate_kernel = np.ones((dilation_size, dilation_size), np.uint8)
        closed = cv2.dilate(closed, dilate_kernel)
    return closed

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
        elapsed = time.time() - (self.start_time or 0)
        
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
                        img = img.resize((self.target_width, new_height), Image.Resampling.LANCZOS)
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
                        img = img.resize((self.target_width, new_height), Image.Resampling.LANCZOS)
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
                    return None, None, 0.0
                
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
                
                if len(good_matches) < 10:
                    return None, None, 0.0
                
                # Extract matched keypoints
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography
                homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if homography is None:
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
                
                if max_val >= 0.6: 
                    top_left_x = max_loc[0]
                    top_left_y = max_loc[1] + y_start  # Add offset
                    return (top_left_x, top_left_y), max_val, merged_template, images_used
                else:
                    return None, max_val, merged_template, images_used
                        
            except Exception as e:
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
                    
                    # Find position using improved SIFT matching
                    position_result = find_template_position_sift(raw_image, template, search_region)
                    
                    skip_count = 0
                    merged_template = None
                    
                    if position_result and position_result[2] >= 0.7:
                        position, transformed_corners, confidence = position_result
                        method = "SIFT"
                        
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
                        # Fallback to template matching
                        fallback_result = find_template_position_template_matching(
                            raw_image, 
                            template, 
                            search_region,
                            small_images=images,
                            current_index=i
                        )

                        if fallback_result:
                            confidence = fallback_result[1]
                            position = fallback_result[0]
                            merged_template = fallback_result[2]
                            skip_count = fallback_result[3] if len(fallback_result) > 3 else 0

                            if confidence >= 0.6:
                                method = "Template matching"
                                if merged_template is not None:
                                    template = merged_template
                                    merged_mask_array = np.ones((merged_template.shape[0], merged_template.shape[1]), dtype=np.uint8) * 255
                                    mask_array = merged_mask_array
                                top_left = position
                                width, height = template.shape[1], template.shape[0]
                                resized_template = template
                                resized_mask = mask_array

                            elif 0.2 <= confidence < 0.6:
                                method = "Concat-below"
                                x = 0
                                y = current_y
                                width, height = template.shape[1], template.shape[0]
                                resized_template = template
                                resized_mask = mask_array
                                top_left = (x, y)

                            else:
                                i += 1
                                continue
                        else:
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

    def create_final_results_with_raw_mask(self, dir_to_process=None, kernel_size=100, dilation_size=50):
        """Step 5+6 Combined: Create final results using raw mask closure/dilation to copy VN regions to raw image"""
        print("\n=== Creating Final Results with Raw Mask Processing ===")
        
        # Set up paths
        vn_path = os.path.join(self.base_path, 'vn')
        raw_path = os.path.join(self.base_path, 'raw')
        raw_mask_path = os.path.join(self.base_path, 'raw', 'mask')
        result_path = os.path.join(self.base_path, 'result')
        
        # Find files to process
        vn_files = [f for f in os.listdir(vn_path) if f.endswith('.png')]
        
        # Filter by dir_to_process if specified
        if dir_to_process:
            print(f"Processing only directory: {dir_to_process}")
            vn_files = [f for f in vn_files if f.startswith(f"{dir_to_process}.")]
            
        if not vn_files:
            print("⚠️ No VN files found. Skipping final results generation.")
            return
            
        self.start_progress(len(vn_files))
        
        def process_single_file(filename):
            try:
                # Define file paths
                vn_image_path = os.path.join(vn_path, filename)
                raw_image_path = os.path.join(raw_path, filename)
                raw_mask_path_file = os.path.join(raw_mask_path, filename)
                result_path_file = os.path.join(result_path, filename)
                
                # Check if all required files exist
                if not os.path.exists(vn_image_path):
                    print(f"⚠️ VN image not found: {vn_image_path}")
                    self.update_progress()
                    return
                    
                if not os.path.exists(raw_image_path):
                    print(f"⚠️ Raw image not found: {raw_image_path}")
                    self.update_progress()
                    return
                    
                if not os.path.exists(raw_mask_path_file):
                    print(f"⚠️ Raw mask not found: {raw_mask_path_file}")
                    self.update_progress()
                    return
                
                # Load images and mask
                vn_image = cv2.imread(vn_image_path)
                raw_image = cv2.imread(raw_image_path)
                raw_mask = cv2.imread(raw_mask_path_file, cv2.IMREAD_GRAYSCALE)
                
                if vn_image is None or raw_image is None or raw_mask is None:
                    print(f"⚠️ Failed to read images for {filename}")
                    self.update_progress()
                    return
                
                # Ensure images have same dimensions
                if vn_image.shape[:2] != raw_image.shape[:2]:
                    print(f"⚠️ Image dimensions don't match for {filename}")
                    self.update_progress()
                    return
                
                # Convert mask to binary
                raw_mask_bin = (raw_mask > 0).astype(np.uint8)
                
                # Apply closure and dilation to raw mask
                processed_mask = closure_mask(raw_mask_bin, kernel_size, dilation_size)
                
                # Find connected components in processed mask
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_mask, connectivity=8)
                
                # Create result image (start with raw image)
                result_image = raw_image.copy()
                
                # Process each connected component
                regions_copied = 0
                for label in range(1, num_labels):  # Skip background (label 0)
                    # Get region coordinates
                    x = stats[label, cv2.CC_STAT_LEFT]
                    y = stats[label, cv2.CC_STAT_TOP]
                    w = stats[label, cv2.CC_STAT_WIDTH]
                    h = stats[label, cv2.CC_STAT_HEIGHT]
                    
                    # Ensure coordinates are within image bounds
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, raw_image.shape[1] - x)
                    h = min(h, raw_image.shape[0] - y)
                    
                    if w <= 0 or h <= 0:
                        continue
                    
                    # Extract region from processed mask
                    region_mask = processed_mask[y:y+h, x:x+w]
                    
                    # Extract corresponding region from VN image
                    vn_region = vn_image[y:y+h, x:x+w]
                    
                    # Extract corresponding region from raw image
                    raw_region = raw_image[y:y+h, x:x+w]
                    
                    # Create normalized mask for blending
                    region_mask_norm = region_mask.astype(np.float32) / 255.0
                    region_mask_norm = np.expand_dims(region_mask_norm, axis=2)  # Add channel dimension
                    
                    # Blend VN region with raw region using the mask
                    blended_region = (vn_region * region_mask_norm + 
                                    raw_region * (1 - region_mask_norm)).astype(np.uint8)
                    
                    # Paste blended region back to result image
                    result_image[y:y+h, x:x+w] = blended_region
                    
                    regions_copied += 1
                
                # Save result
                cv2.imwrite(result_path_file, result_image)
                print(f"✅ Created final result: {filename} (copied {regions_copied} regions)")
                self.update_progress()
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                import traceback
                traceback.print_exc()
                self.update_progress()
        
        # Process files in parallel
        pool = ThreadPool(self.thread_count)
        try:
            pool.map(process_single_file, vn_files)
        finally:
            pool.close()
            pool.join()
        
        print("Final results generation with raw mask processing completed!")
        
    def run_pipeline(self, dir_to_process=None):
        """Run the complete pipeline"""
        print("\n🚀 Starting Manga Processing Pipeline 🚀")
        print(f"Base path: {self.base_path}")
        print(f"Using {self.thread_count} threads\n")
        
        if dir_to_process:
            print(f"Processing only directory: {dir_to_process}")
        
        start_time = time.time()
        
        # Execute all steps
        self.merge_raw_images(dir_to_process)
        self.merge_raw_masks(dir_to_process)
        self.resize_vn_images(dir_to_process)
        self.merge_vn_images_vertically(dir_to_process)
        self.create_final_results_with_raw_mask(dir_to_process)  # New combined step 5+6
        
        elapsed = time.time() - start_time
        print(f"\n✅ Pipeline completed in {elapsed:.1f} seconds!")
        print(f"Final results are available in: {self.paths['result']}")

# Main execution
if __name__ == "__main__":
    base_path = r"e:\Manwa\reincarnation-path-of-the-underworld-king"
    
    processor = MangaProcessor(base_path, threads=12)
    
    dir_to_process = "c1"
    
    # Run pipeline with focus dir
    processor.run_pipeline(dir_to_process)
    
    # processor.run_pipeline()