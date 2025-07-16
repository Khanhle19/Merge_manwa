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
        """Step 4: Merge VN images vertically with template matching"""
        print("\n=== Merging VN Images Vertically ===")
        
        # Get directories to process
        if not os.path.exists(self.paths['vn1']):
            print("⚠️ VN1 directory not found. Skipping vertical merging.")
            return
            
        if dir_to_process:
            print(f"Processing only directory: {dir_to_process}")
            dirs = [dir_to_process]
        else:
            dirs = os.listdir(self.paths['vn1'])
            
        self.start_progress(len(dirs))
        
        # Define worker function
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
                    key=lambda x: x.lower()
                )
                
                if not files:
                    print(f"⚠️ No images found for {name}")
                    self.update_progress()
                    return
                
                # Load images and masks
                original_images = []
                original_masks = []

                for filename in files:
                    path = os.path.join(folder_path, filename)
                    mask_path = os.path.join(folder_path, 'mask', filename)
                    
                    # Try different extensions for mask if the exact match doesn't exist
                    if not os.path.exists(mask_path):
                        base_name = os.path.splitext(filename)[0]
                        mask_path = os.path.join(folder_path, 'mask', f"{base_name}.png")
                        if not os.path.exists(mask_path):
                            print(f"⚠️ Mask not found for {filename}")
                            continue
                    
                    try:
                        with Image.open(path) as img:
                            img = img.convert("RGB")
                            original_images.append(img.copy())
                        with Image.open(mask_path) as m:
                            m = m.convert("L")
                            original_masks.append(m.copy())
                    except Exception as e:
                        print(f"⚠️ Error opening {filename} or its mask: {e}")

                if not original_images:
                    print(f"⚠️ No valid image-mask pairs found for {name}")
                    self.update_progress()
                    return

                # NEW CODE: Check and merge images that are too small (height < 1000 pixels)
                images = []
                masks = []
                i = 0
                
                while i < len(original_images):
                    current_image = original_images[i]
                    current_mask = original_masks[i]
                    
                    # If image height is less than 1000 pixels, try to merge with next images
                    if current_image.height < 1000 and i + 1 < len(original_images):
                        merged_height = current_image.height
                        merge_candidates = [i]
                        j = i + 1
                        
                        # Find consecutive images to merge until height >= 1000 pixels
                        while merged_height < 1000 and j < len(original_images):
                            merged_height += original_images[j].height
                            merge_candidates.append(j)
                            j += 1
                            
                        if merged_height >= 1000 or len(merge_candidates) > 1:
                            # Create new merged image and mask
                            merged_img = Image.new("RGB", (self.target_width, merged_height))
                            merged_mask = Image.new("L", (self.target_width, merged_height))
                            
                            # Paste images and masks
                            y_offset = 0
                            for idx in merge_candidates:
                                img = original_images[idx]
                                msk = original_masks[idx]
                                merged_img.paste(img, (0, y_offset))
                                merged_mask.paste(msk, (0, y_offset))
                                y_offset += img.height
                            
                            # Add merged image and mask to lists
                            images.append(merged_img)
                            masks.append(merged_mask)
                            
                            print(f"✅ Merged {len(merge_candidates)} small images for better template matching in {name}")
                            i = merge_candidates[-1] + 1
                        else:
                            # Just add the current image if no merging occurred
                            images.append(current_image)
                            masks.append(current_mask)
                            i += 1
                    else:
                        # Add image and mask as is if height >= 1000 pixels
                        images.append(current_image)
                        masks.append(current_mask)
                        i += 1

                # Create merged images
                raw_image = None
                raw_height = 0
                raw_width = self.target_width
                if os.path.exists(raw_path):
                    raw_image = cv2.imread(raw_path, cv2.IMREAD_COLOR)
                    if raw_image is None:
                        print(f"⚠️ Could not read raw image: {raw_path}")
                    else:
                        raw_height, raw_width, _ = raw_image.shape

                # Create new images for merging
                merged = Image.new("RGB", (raw_width, raw_height), color=(255, 255, 255))
                merged_mask = Image.new("L", (raw_width, raw_height), color=0)

                # Place images based on template matching
                y = 0
                first_image_pasted = False

                for i, img in enumerate(images):
                    # Convert PIL image to OpenCV format for template matching
                    cv_img = np.array(img)
                    cv_img = cv_img[:, :, ::-1].copy()  # RGB to BGR
                    
                    # If we have a valid raw image, use template matching
                    if raw_image is not None:
                        try:
                            template = cv_img
                            if template is not None:
                                h, w = template.shape[:2]
                                
                                if i == 0:
                                    result = cv2.matchTemplate(raw_image, template, cv2.TM_CCOEFF_NORMED) 
                                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                                    
                                    if max_val > 0.6 and max_loc[1] < 5000:
                                        y = max_loc[1]
                                    else:
                                        continue
                                else:
                                    search_found = False
                                    
                                    roi_start = max(0, y - 3000)
                                    roi_end = min(raw_height - h, y + 5000)
                                    
                                    if roi_end > roi_start and roi_end < raw_height:
                                        roi = raw_image[roi_start:roi_end, 0:raw_width]
                                        
                                        # Only perform template matching if ROI is large enough
                                        if roi.shape[0] > 0 and roi.shape[1] > 0:
                                            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
                                            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                                            
                                            if max_val > 0.6:
                                                y = roi_start + max_loc[1]
                                                search_found = True
                                    
                                    if not search_found:
                                        result = cv2.matchTemplate(raw_image, template, cv2.TM_CCOEFF_NORMED)
                                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                                        
                                        if max_val > 0.75 and max_loc[1] > 0 and max_loc[1] < raw_height:
                                            y = max_loc[1]

                        except Exception as e:
                            print(f"⚠️ Template matching error: {e}")
                    
                    # Paste images
                    merged.paste(images[i], (0, y))
                    merged_mask.paste(masks[i], (0, y))

                    if i != 0 and not first_image_pasted:  # paste first image if it wasn't placed
                        merged.paste(images[0], (0, y - images[0].height))
                        merged_mask.paste(masks[0], (0, y - masks[0].height))
                        first_image_pasted = True

                    y += images[i].height
                
                # Save results
                merged.save(output_name, quality=95)
                merged_mask.save(output_mask, quality=95)
                print(f"✅ Created merged VN image and mask: {name}.png")
                self.update_progress()
            except Exception as e:
                print(f"❌ Error merging VN images for {dir}: {e}")
                self.update_progress()
        
        # Process in parallel
        pool = ThreadPool(self.thread_count)
        try:
            pool.map(merge_images_vertically, dirs)
        finally:
            pool.close()
            pool.join()
        
        print("VN image vertical merging completed!")


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

    def create_intersection_masks(self, dir_to_process=None):
        """Step 5: Create intersection masks using GIS"""
        print("\n=== Creating Intersection Masks ===")
        
        maskvn_path = os.path.join(self.base_path, 'vn', 'mask')
        maksraw_path = os.path.join(self.base_path, 'raw', 'mask')
        mask_path = os.path.join(self.base_path, 'mask')
        
        for path in [maskvn_path, maksraw_path, mask_path]:
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
        
        # Tạo danh sách tham số cho mỗi tác vụ
        args = [(path, maskvn_path, maksraw_path, mask_path) for path in lists]
        
        # Xử lý song song
        pool = ProcessPool(self.thread_count)
        try:
            results = pool.starmap(MangaProcessor.create_mask, args)
            # for _ in results:
            #     self.update_progress()
        finally:
            pool.close()
            pool.join()

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
        self.merge_raw_images(dir_to_process)
        self.merge_raw_masks(dir_to_process)
        self.resize_vn_images(dir_to_process)
        self.merge_vn_images_vertically(dir_to_process)
        self.create_intersection_masks(dir_to_process)
        self.generate_final_results(dir_to_process)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Pipeline completed in {elapsed:.1f} seconds!")
        print(f"Final results are available in: {self.paths['result']}")

# Main execution
if __name__ == "__main__":
    base_path = r"e:\Manwa\infinite-mage-941"
    
    processor = MangaProcessor(base_path, threads=12)
    
    dir_to_process = "c105"
    
    # Chạy pipeline với thư mục đã chỉ định
    processor.run_pipeline(dir_to_process)
    
    # processor.run_pipeline()