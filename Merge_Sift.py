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

def closure_mask(mask, kernel_size=30, dilation_size=20):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if dilation_size > 0:
        dilate_kernel = np.ones((dilation_size, dilation_size), np.uint8)
        closed = cv2.dilate(closed, dilate_kernel)
    return closed

def pipeline_mask_logic(mask_vn_path, mask_raw_path, output_path, kernel_size=30, dilation_size=20):
    mask_vn = np.array(Image.open(mask_vn_path).convert('L'))
    mask_raw = np.array(Image.open(mask_raw_path).convert('L'))

    mask_vn_bin = (mask_vn > 0).astype(np.uint8)
    mask_raw_bin = (mask_raw > 0).astype(np.uint8)

    if mask_vn_bin.shape != mask_raw_bin.shape:
        mask_raw_bin = cv2.resize(mask_raw_bin, (mask_vn_bin.shape[1], mask_vn_bin.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Morphological processing để tạo các vùng liên thông lớn
    closure_raw = closure_mask(mask_raw_bin, kernel_size, dilation_size + 50)
    closure_vn = closure_mask(mask_vn_bin, kernel_size, dilation_size)

    # 🎯 LOGIC MỚI: Tạo union để có tất cả vùng tiềm năng
    union_mask = np.logical_or(closure_raw, closure_vn).astype(np.uint8)
    
    # Tìm connected components từ union
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(union_mask, connectivity=8)
    
    # 🔑 LOGIC CHÍNH: Giữ toàn bộ component nếu có intersection
    final_mask = np.zeros_like(union_mask)
    valid_components = 0
    
    # print(f"Found {num_labels-1} connected components")
    
    for label in range(1, num_labels):  # Skip background (label 0)
        # Lấy toàn bộ vùng của component này
        component_region = (labels == label)
        
        # Kiểm tra xem trong component này có pixel nào thuộc cả 2 masks không
        intersection_in_component = np.logical_and(
            closure_raw[component_region], 
            closure_vn[component_region]
        )
        
        # 🔑 ĐIỀU KIỆN: Nếu có ít nhất 1 pixel intersection thì giữ TOÀN BỘ component
        if intersection_in_component.sum() > 0:
            # Giữ toàn bộ component (không chỉ phần intersection)
            final_mask[component_region] = 1
            valid_components += 1
            print(f"✅ Kept entire component {label}: has {intersection_in_component.sum()} intersection pixels")
        else:
            print(f"❌ Rejected component {label}: no intersection pixels")
    
    print(f"Kept {valid_components} out of {num_labels-1} components")
    
    # Lưu kết quả
    Image.fromarray((final_mask * 255).astype(np.uint8)).save(output_path)

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
        
        # # Debug: Check critical paths and file counts
        # print(f"🔍 Path verification:")
        # critical_paths = ['raw2_inpainted', 'raw2_mask', 'raw1']
        # for path_name in critical_paths:
        #     path = self.paths[path_name]
        #     exists = os.path.exists(path)
        #     file_count = 0
        #     sample_files = []
            
        #     if exists:
        #         try:
        #             all_files = os.listdir(path)
        #             png_files = [f for f in all_files if f.endswith('.png')]
        #             file_count = len(png_files)
        #             sample_files = png_files[:3]
        #         except:
        #             file_count = "Error reading"
                    
        #     print(f"   {path_name}: {path}")
        #     print(f"     Exists: {exists}, PNG files: {file_count}")
        #     if sample_files:
        #         print(f"     Sample files: {sample_files}")
        # print()

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
                gpath = os.path.join(self.paths['raw2'], f"{dir}-*.jpg")
                image_files = sorted(glob.glob(gpath), key=MangaProcessor.natural_sort_key)
                
                if not image_files:
                    print(f"⚠️ No images found for {dir} at path: {gpath}")
                    # Debug: Show what files actually exist
                    inpainted_dir = self.paths['raw2']
                    if os.path.exists(inpainted_dir):
                        all_files = os.listdir(inpainted_dir)
                        matching_files = [f for f in all_files if f.startswith(f"{dir}-") and f.endswith('.jpg')]
                        print(f"   Directory exists with {len(all_files)} total files")
                        print(f"   Found {len(matching_files)} files matching pattern '{dir}-*.jpg'")
                        if matching_files:
                            print(f"   Sample files: {matching_files[:5]}")
                    else:
                        print(f"   Directory {inpainted_dir} does not exist")
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
                        # Remove width restriction - accept all valid images
                        # Original logic: if x > 1000: continue
                        # New logic: accept all images
                        ratio = self.target_width / x
                        y += int(img.size[1] * ratio)
                        valid_files.append(path)
                        print(f"✅ Valid image: {os.path.basename(path)} ({x}x{img.size[1]})")
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
                    print(f"⚠️ No mask files found for {dir} at path: {gpath}")
                    # Debug: Show what files actually exist
                    mask_dir = self.paths['raw2_mask']
                    if os.path.exists(mask_dir):
                        all_files = os.listdir(mask_dir)
                        matching_files = [f for f in all_files if f.startswith(f"{dir}-") and f.endswith('.png')]
                        print(f"   Directory exists with {len(all_files)} total files")
                        print(f"   Found {len(matching_files)} files matching pattern '{dir}-*.png'")
                        if matching_files:
                            print(f"   Sample files: {matching_files[:5]}")
                    else:
                        print(f"   Directory {mask_dir} does not exist")
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
                        # Remove width restriction - accept all valid masks
                        # Original logic: if x > 1000: continue  
                        # New logic: accept all masks
                        ratio = self.target_width / x
                        y += int(img.size[1] * ratio)
                        valid_files.append(path)
                        print(f"✅ Valid mask: {os.path.basename(path)} ({x}x{img.size[1]})")
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
        
        def find_template_position_template_matching(raw_image, template, search_region, small_images=None, small_masks=None, current_index=None):
            try:
                # Extract the search region
                y_start, y_end = search_region
                y_start = max(0, y_start)
                y_end = min(raw_image.shape[0], y_end)
                region = raw_image[y_start:y_end, :].copy()
                
                # Check if template is too small (height < 800) and we have additional images
                merged_template = None
                merged_mask = None
                images_used = 0 

                if template.shape[0] < 800 and small_images is not None and current_index is not None:
                    print(f"Template height ({template.shape[0]}) < 800, attempting to merge with next images")
                    merged_height = template.shape[0]
                    merge_candidates = [template]
                    mask_candidates = [small_masks[current_index] if small_masks else None]
                    next_idx = current_index + 1

                    # Find consecutive images to merge until height >= 800 pixels
                    while merged_height < 800 and next_idx < len(small_images):
                        next_image = small_images[next_idx]
                        next_mask = small_masks[next_idx] if small_masks and next_idx < len(small_masks) else None
                        merged_height += next_image.shape[0]
                        merge_candidates.append(next_image)
                        mask_candidates.append(next_mask)
                        images_used += 1  
                        next_idx += 1
                        
                    if len(merge_candidates) > 1:
                        # Create merged template
                        merged_template = np.zeros((merged_height, template.shape[1], 3), dtype=np.uint8)
                        merged_mask = np.zeros((merged_height, template.shape[1]), dtype=np.uint8)
                        y_offset = 0
                        
                        for idx, img in enumerate(merge_candidates):
                            corresponding_mask = mask_candidates[idx]
                            
                            # Check if image is wider than merged template
                            if img.shape[1] > merged_template.shape[1]:
                                scale_factor = merged_template.shape[1] / img.shape[1]
                                new_width = merged_template.shape[1]
                                new_height = int(img.shape[0] * scale_factor)
                                img = cv2.resize(img, (new_width, new_height))
                                if corresponding_mask is not None:
                                    corresponding_mask = cv2.resize(corresponding_mask, (new_width, new_height))
                            
                            # Place image in merged template
                            h = img.shape[0]
                            merged_template[y_offset:y_offset+h, 0:img.shape[1]] = img
                            
                            # Place corresponding mask in merged mask
                            if corresponding_mask is not None:
                                merged_mask[y_offset:y_offset+h, 0:corresponding_mask.shape[1]] = corresponding_mask
                            else:
                                # If no mask available, create white mask for this segment
                                merged_mask[y_offset:y_offset+h, 0:img.shape[1]] = 255
                            
                            y_offset += h
                        
                        print(f"Merged {len(merge_candidates)} images and masks into template of size {merged_template.shape[1]}x{merged_template.shape[0]}")
                        template = merged_template                # Resize template if needed
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
                    print(f"Template matching found position at ({top_left_x}, {top_left_y}) with confidence {max_val:.3f}")
                    return (top_left_x, top_left_y), max_val, merged_template, merged_mask, images_used
                else:
                    print(f"Template matching confidence too low: {max_val:.3f}")
                    return None, max_val, merged_template, merged_mask, images_used
                        
            except Exception as e:
                print(f"⚠️ Template matching error: {e}")
                return None, 0.0, None, None, 0
        
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
                        search_region = (0, min(20000, raw_height))
                    else:
                        search_region = (max(0, current_y - 18000), min(raw_height, current_y + 15000))
                    
                    print(f"Searching for {files[i]} in region y={search_region[0]} to y={search_region[1]}")
                    
                    # Find position using improved SIFT matching
                    position_result = find_template_position_sift(raw_image, template, search_region)
                    
                    skip_count = 0
                    merged_template = None
                    
                    if position_result and position_result[2] >= 0.7:
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
                            small_masks=masks,
                            current_index=i
                        )

                        if fallback_result:
                            confidence = fallback_result[1]
                            position = fallback_result[0]
                            merged_template = fallback_result[2]
                            merged_mask_from_template = fallback_result[3]
                            skip_count = fallback_result[4] if len(fallback_result) > 4 else 0

                            if confidence >= 0.4:
                                method = "Template matching"
                                print(f"✅ Template matching for {files[i]}: position {position}, confidence: {confidence:.3f}")
                                if merged_template is not None:
                                    print(f"Using merged template for positioning")
                                    template = merged_template
                                    # Use merged mask instead of blank white mask
                                    if merged_mask_from_template is not None:
                                        mask_array = merged_mask_from_template
                                        print(f"Using merged mask with {len([i for i in range(skip_count+1)])} masks combined")
                                    else:
                                        merged_mask_array = np.ones((merged_template.shape[0], merged_template.shape[1]), dtype=np.uint8) * 255
                                        mask_array = merged_mask_array
                                        print(f"No merged mask available, using blank white mask")
                                top_left = position
                                width, height = template.shape[1], template.shape[0]
                                resized_template = template
                                resized_mask = mask_array

                            elif 0.3 <= confidence < 0.4:
                                method = "Concat-below"
                                print(f"⚠️ Confidence ({confidence:.3f}) low, concat below previous image.")
                                
                                # Check if we have merged template from failed template matching
                                if merged_template is not None:
                                    print(f"Using merged template for concat-below positioning")
                                    template = merged_template
                                    # Use merged mask instead of single mask
                                    if merged_mask_from_template is not None:
                                        mask_array = merged_mask_from_template
                                        print(f"Using merged mask with {skip_count+1} masks combined for concat-below")
                                    else:
                                        merged_mask_array = np.ones((merged_template.shape[0], merged_template.shape[1]), dtype=np.uint8) * 255
                                        mask_array = merged_mask_array
                                        print(f"No merged mask available for concat-below, using blank white mask")
                                
                                x = 0
                                y = current_y
                                width, height = template.shape[1], template.shape[0]
                                resized_template = template
                                resized_mask = mask_array
                                top_left = (x, y)

                            else:
                                print(f"⚠️ Could not find position for {files[i]} with sufficient confidence")
                                i += 1
                                continue
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

    def create_intersection_masks(self, dir_to_process=None, kernel_size=30, dilation_size=20):
        """Step 5: Pipeline-based closure/union/exclusion mask logic with dilation"""
        print("\n=== Creating Intersection Masks (Pipeline Logic + Dilation) ===")
        
        maskvn_path = os.path.join(self.base_path, 'vn', 'mask')
        maksraw_path = os.path.join(self.base_path, 'raw', 'mask')
        mask_path = os.path.join(self.base_path, 'mask')
        os.makedirs(mask_path, exist_ok=True)
        
        # Get mask files to process
        lists = [f for f in os.listdir(maskvn_path) if f.endswith('.png')]
        if dir_to_process:
            lists = [f"{dir_to_process}.png"] if f"{dir_to_process}.png" in lists else []
        
        if not lists:
            print("⚠️ No mask files found")
            return
        
        self.start_progress(len(lists))
        
        def worker(f):
            mask_vn_file = os.path.join(maskvn_path, f)
            mask_raw_file = os.path.join(maksraw_path, f)
            output_file = os.path.join(mask_path, f)
            if not os.path.exists(mask_raw_file):
                print(f"⚠️ Raw mask not found for {f}")
                self.update_progress()
                return
            try:
                pipeline_mask_logic(mask_vn_file, mask_raw_file, output_file, kernel_size, dilation_size)
                print(f"✅ Created intersection mask: {f}")
            except Exception as e:
                print(f"❌ Error processing {f}: {e}")
            self.update_progress()
        
        from multiprocessing.dummy import Pool as ThreadPool
        with ThreadPool(self.thread_count) as pool:
            pool.map(worker, lists)
        
        print("Mask intersection (pipeline logic + dilation) completed!")
        
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
        self.merge_vn_images_vertically(dir_to_process)
        self.create_intersection_masks(dir_to_process)
        self.generate_final_results(dir_to_process)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Pipeline completed in {elapsed:.1f} seconds!")
        print(f"Final results are available in: {self.paths['result']}")

# Main execution
if __name__ == "__main__":
    base_path = r"f:\manhwa\Nhiem_vu_doi_that"
    
    processor = MangaProcessor(base_path, threads=12)
    
    dir_to_process = "c1" 
    
    processor.run_pipeline(dir_to_process)
    
    # processor.run_pipeline()