import sys
import os
import json
import re
import random
import numpy as np
from multiprocessing import Pool, cpu_count
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QSpinBox, QCheckBox, QComboBox, QProgressBar, 
                             QTextEdit, QFileDialog, QGroupBox, QGridLayout, 
                             QMessageBox, QTabWidget, QRadioButton, QListWidget,
                             QListWidgetItem, QTreeWidget, QTreeWidgetItem,
                             QSplitter, QScrollArea)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QIcon, QPixmap
import threading
import time
import cv2

def compare_images_sift_worker(task):
    """Worker function for multiprocessing SIFT comparison"""
    try:
        # Load images as grayscale
        img1 = cv2.imread(task['sample_path'], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(task['img2_path'], cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            return 0.0
        
        # Initialize SIFT
        sift = cv2.SIFT_create(nfeatures=task['max_keypoints'])
        
        # Extract keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return 0.0
        
        # FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        # Calculate similarity score
        if len(kp1) == 0 or len(kp2) == 0:
            return 0.0
            
        similarity = len(good_matches) / max(len(kp1), len(kp2))
        return similarity
        
    except Exception as e:
        return 0.0

class FolderRenamerWorker(QThread):
    """Worker thread để đổi tên folder không block UI"""
    progress_updated = pyqtSignal(int)
    log_updated = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_cancelled = False
    
    def cancel(self):
        self.is_cancelled = True
    
    def run(self):
        try:
            self.process_folders()
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")
    
    def process_folders(self):
        config = self.config
        
        # Validation
        if not os.path.exists(config['target_directory']):
            self.finished.emit(False, "Target directory not found!")
            return
        
        # Find folders
        try:
            items = os.listdir(config['target_directory'])
        except Exception as e:
            self.finished.emit(False, f"Error reading directory: {str(e)}")
            return
        
        # Filter folders only
        folders = [item for item in items if os.path.isdir(os.path.join(config['target_directory'], item))]
        
        if config['filter_pattern']:
            # Apply filter pattern
            pattern = config['filter_pattern']
            if config['use_regex']:
                try:
                    regex = re.compile(pattern)
                    folders = [f for f in folders if regex.search(f)]
                except re.error as e:
                    self.finished.emit(False, f"Invalid regex pattern: {str(e)}")
                    return
            else:
                # Simple wildcard pattern
                pattern = pattern.replace('*', '.*').replace('?', '.')
                try:
                    regex = re.compile(pattern)
                    folders = [f for f in folders if regex.search(f)]
                except re.error:
                    folders = [f for f in folders if pattern in f]
        
        if not folders:
            self.finished.emit(False, "No folders found matching the criteria!")
            return
        
        self.log_updated.emit(f"Found {len(folders)} folders to rename")
        
        # Sort folders
        if config['sort_naturally']:
            folders = self.natural_sort(folders)
        else:
            folders = sorted(folders)
        
        # Process renaming
        success_count = 0
        error_count = 0
        renamed_pairs = []
        
        for i, folder_name in enumerate(folders):
            if self.is_cancelled:
                break
                
            old_path = os.path.join(config['target_directory'], folder_name)
            new_name = self.generate_new_name(folder_name, i, config)
            new_path = os.path.join(config['target_directory'], new_name)
            
            try:
                if old_path != new_path and not os.path.exists(new_path):
                    if config['preview_mode']:
                        self.log_updated.emit(f"Preview: {folder_name} -> {new_name}")
                        renamed_pairs.append((folder_name, new_name))
                    else:
                        os.rename(old_path, new_path)
                        self.log_updated.emit(f"Renamed: {folder_name} -> {new_name}")
                        renamed_pairs.append((folder_name, new_name))
                    success_count += 1
                elif old_path == new_path:
                    self.log_updated.emit(f"Skipped (no change): {folder_name}")
                else:
                    self.log_updated.emit(f"Error: Target exists: {new_name}")
                    error_count += 1
            except Exception as e:
                self.log_updated.emit(f"Error renaming {folder_name}: {str(e)}")
                error_count += 1
            
            # Update progress
            progress = int((i + 1) / len(folders) * 100)
            self.progress_updated.emit(progress)
        
        # Finish
        mode_text = "Previewed" if config['preview_mode'] else "Renamed"
        summary = f"Completed! {mode_text}: {success_count}/{len(folders)}, Errors: {error_count}"
        self.finished.emit(True, summary)
    
    def natural_sort(self, folders):
        """Enhanced natural sorting with decimal support (c1, c1.5, c2, c10 instead of c1, c10, c1.5, c2)"""
        def natural_key(text):
            # Split text into parts and handle numbers with decimals
            parts = []
            for part in re.split(r'(\d+(?:\.\d+)?)', text):
                if re.match(r'\d+(?:\.\d+)?$', part):
                    # Convert to float for proper decimal sorting
                    parts.append(float(part))
                else:
                    # Keep text parts as lowercase
                    parts.append(part.lower())
            return parts
        return sorted(folders, key=natural_key)
    
    def generate_new_name(self, old_name, index, config):
        """Generate new folder name based on configuration with improved logic"""
        if config['rename_method'] == 'pattern':
            # Pattern-based renaming
            new_name = config['name_pattern']
            
            # Replace basic placeholders
            new_name = new_name.replace('{index}', str(index + config['start_number']))
            new_name = new_name.replace('{index:02d}', f"{index + config['start_number']:02d}")
            new_name = new_name.replace('{index:03d}', f"{index + config['start_number']:03d}")
            new_name = new_name.replace('{index:04d}', f"{index + config['start_number']:04d}")
            new_name = new_name.replace('{original}', old_name)
            
            # Enhanced number extraction with decimal support
            # Extract all numbers including decimals (e.g., 192.5, 15.2, etc.)
            decimal_numbers = re.findall(r'\d+(?:\.\d+)?', old_name)
            # Extract only integer numbers
            integer_numbers = re.findall(r'\d+', old_name)
            
            # Handle decimal numbers
            if decimal_numbers:
                new_name = new_name.replace('{number}', decimal_numbers[0])
                new_name = new_name.replace('{decimal}', decimal_numbers[0])
                if len(decimal_numbers) > 1:
                    new_name = new_name.replace('{number2}', decimal_numbers[1])
                    new_name = new_name.replace('{decimal2}', decimal_numbers[1])
            
            # Handle integer-only placeholders
            if integer_numbers:
                new_name = new_name.replace('{integer}', integer_numbers[0])
                if len(integer_numbers) > 1:
                    new_name = new_name.replace('{integer2}', integer_numbers[1])
            
            # Smart chapter extraction for common patterns
            chapter_patterns = [
                r'[Cc]h(?:apter)?\s*(\d+(?:\.\d+)?)',  # Ch 192.5, Chapter 192.5
                r'[Cc]hương\s*(\d+(?:\.\d+)?)',        # Chương 192.5
                r'[Tt]ập\s*(\d+(?:\.\d+)?)',           # Tập 192.5
                r'(\d+(?:\.\d+)?)$',                   # Number at end: "Something 192.5"
                r'^(\d+(?:\.\d+)?)',                   # Number at start: "192.5 Something"
            ]
            
            chapter_number = None
            for pattern in chapter_patterns:
                match = re.search(pattern, old_name)
                if match:
                    chapter_number = match.group(1)
                    break
            
            if chapter_number:
                new_name = new_name.replace('{chapter}', chapter_number)
                new_name = new_name.replace('{ch}', chapter_number)
                
                # Zero-padded chapter numbers
                try:
                    if '.' in chapter_number:
                        # Handle decimal chapters (192.5 -> 192.5, not zero-padded)
                        new_name = new_name.replace('{chapter:03d}', chapter_number)
                        new_name = new_name.replace('{ch:03d}', chapter_number)
                    else:
                        # Integer chapters can be zero-padded
                        ch_int = int(chapter_number)
                        new_name = new_name.replace('{chapter:03d}', f"{ch_int:03d}")
                        new_name = new_name.replace('{ch:03d}', f"{ch_int:03d}")
                        new_name = new_name.replace('{chapter:02d}', f"{ch_int:02d}")
                        new_name = new_name.replace('{ch:02d}', f"{ch_int:02d}")
                except ValueError:
                    # If conversion fails, use original
                    new_name = new_name.replace('{chapter:03d}', chapter_number)
                    new_name = new_name.replace('{ch:03d}', chapter_number)
                    new_name = new_name.replace('{chapter:02d}', chapter_number)
                    new_name = new_name.replace('{ch:02d}', chapter_number)
            
            return new_name
            
        elif config['rename_method'] == 'replace':
            # Find and replace
            if config['use_regex']:
                try:
                    return re.sub(config['find_text'], config['replace_text'], old_name)
                except re.error:
                    return old_name
            else:
                return old_name.replace(config['find_text'], config['replace_text'])
                
        elif config['rename_method'] == 'case':
            # Case conversion
            if config['case_option'] == 'lower':
                return old_name.lower()
            elif config['case_option'] == 'upper':
                return old_name.upper()
            elif config['case_option'] == 'title':
                return old_name.title()
            elif config['case_option'] == 'capitalize':
                return old_name.capitalize()
        
        return old_name

class DuplicateDetectionWorker(QThread):
    """Worker thread để phát hiện duplicate không block UI"""
    progress_updated = pyqtSignal(int)
    log_updated = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    duplicate_found = pyqtSignal(str, str, str, str, float)  # folder1, img1, folder2, img2, similarity
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_cancelled = False
        self.sift = None
        
    def cancel(self):
        self.is_cancelled = True
    
    def run(self):
        try:
            self.detect_duplicates()
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")
    
    def detect_duplicates(self):
        config = self.config
        
        # Validation
        if not os.path.exists(config['target_directory']):
            self.finished.emit(False, "Target directory not found!")
            return
        
        # Initialize SIFT
        try:
            self.sift = cv2.SIFT_create(nfeatures=config['max_keypoints'])
        except Exception as e:
            self.finished.emit(False, f"Failed to initialize SIFT: {str(e)}")
            return
        
        # Find folders
        try:
            items = os.listdir(config['target_directory'])
            folders = [item for item in items if os.path.isdir(os.path.join(config['target_directory'], item))]
            # Filter folders with renamed pattern (c001, c002, etc.) and also decimal patterns (c1.5, c192.5)
            folders = [f for f in folders if re.match(r'^c\d+(?:\.\d+)?$', f)]
            # Use improved natural sorting
            folders = self.natural_sort_folders(folders)
        except Exception as e:
            self.finished.emit(False, f"Error reading directory: {str(e)}")
            return
        
        if len(folders) < 2:
            self.finished.emit(False, "Need at least 2 folders to compare!")
            return
        
        self.log_updated.emit(f"Found {len(folders)} folders to analyze: {', '.join(folders)}")
        
        # Get image extensions
        extensions = [ext.strip() for ext in config['image_extensions'].split(',')]
        
        # Determine comparison strategy
        if config['compare_adjacent_only']:
            # Compare only adjacent folders (c1 vs c2, c2 vs c3, etc.)
            total_comparisons = len(folders) - 1
            comparison_pairs = [(i, i + 1) for i in range(len(folders) - 1)]
            self.log_updated.emit(f"Comparing {total_comparisons} adjacent folder pairs...")
        else:
            # Compare all pairs of folders
            total_comparisons = len(folders) * (len(folders) - 1) // 2
            comparison_pairs = [(i, j) for i in range(len(folders)) for j in range(i + 1, len(folders))]
            self.log_updated.emit(f"Comparing all {total_comparisons} folder pairs...")
        
        current_comparison = 0
        total_duplicates = 0
        
        # Process comparisons
        for i, j in comparison_pairs:
            if self.is_cancelled:
                return
            
            folder1 = folders[i]
            folder2 = folders[j]
            current_comparison += 1
            
            adjacency_text = "adjacent" if config['compare_adjacent_only'] else "pair"
            self.log_updated.emit(f"\n🔍 Comparing {adjacency_text} folders {folder1} vs {folder2} ({current_comparison}/{total_comparisons})")
            
            # Choose comparison method based on multiprocessing setting
            if config['enable_multiprocessing']:
                duplicates = self.compare_folders_parallel(folder1, folder2, extensions, config)
            else:
                duplicates = self.compare_folders(folder1, folder2, extensions, config)
                
            total_duplicates += len(duplicates)
            
            if duplicates:
                self.log_updated.emit(f"✅ Found {len(duplicates)} potential duplicates")
                for dup in duplicates:
                    self.duplicate_found.emit(folder1, dup['img1'], folder2, dup['img2'], dup['similarity'])
            else:
                self.log_updated.emit("❌ No duplicates found")
            
            # Update progress
            progress = int(current_comparison / total_comparisons * 100)
            self.progress_updated.emit(progress)
        
        # Finish
        summary = f"Detection completed! Found {total_duplicates} potential duplicates across {total_comparisons} folder pairs"
        self.finished.emit(True, summary)
    
    def compare_folders(self, folder1, folder2, extensions, config):
        """Compare two folders for duplicates"""
        folder1_path = os.path.join(config['target_directory'], folder1)
        folder2_path = os.path.join(config['target_directory'], folder2)
        
        # Get images from both folders
        images1 = self.get_images_from_folder(folder1_path, extensions)
        images2 = self.get_images_from_folder(folder2_path, extensions)
        
        if len(images1) < config['min_images_per_folder'] or len(images2) < config['min_images_per_folder']:
            self.log_updated.emit(f"Skipping {folder1} vs {folder2}: insufficient images")
            return []
        
        # Sample random images from folder1
        sample_size = min(config['sample_size'], len(images1))
        sample_images = random.sample(images1, sample_size)
        
        duplicates = []
        
        for sample_img in sample_images:
            if self.is_cancelled:
                break
                
            sample_path = os.path.join(folder1_path, sample_img)
            
            # Compare with all images in folder2
            for img2 in images2:
                if self.is_cancelled:
                    break
                    
                img2_path = os.path.join(folder2_path, img2)
                similarity = self.compare_images_sift(sample_path, img2_path)
                
                if similarity >= config['similarity_threshold']:
                    duplicates.append({
                        'img1': sample_img,
                        'img2': img2,
                        'similarity': similarity
                    })
                    self.log_updated.emit(f"  📸 Match: {sample_img} ↔ {img2} ({similarity:.1%})")
        
        return duplicates
    
    def compare_folders_parallel(self, folder1, folder2, extensions, config):
        """Compare two folders for duplicates using multiprocessing"""
        folder1_path = os.path.join(config['target_directory'], folder1)
        folder2_path = os.path.join(config['target_directory'], folder2)
        
        # Get images from both folders
        images1 = self.get_images_from_folder(folder1_path, extensions)
        images2 = self.get_images_from_folder(folder2_path, extensions)
        
        if len(images1) < config['min_images_per_folder'] or len(images2) < config['min_images_per_folder']:
            self.log_updated.emit(f"Skipping {folder1} vs {folder2}: insufficient images")
            return []
        
        # Sample random images from folder1
        sample_size = min(config['sample_size'], len(images1))
        sample_images = random.sample(images1, sample_size)
        
        # Prepare comparison tasks for multiprocessing
        comparison_tasks = []
        for sample_img in sample_images:
            sample_path = os.path.join(folder1_path, sample_img)
            for img2 in images2:
                img2_path = os.path.join(folder2_path, img2)
                comparison_tasks.append({
                    'sample_img': sample_img,
                    'img2': img2,
                    'sample_path': sample_path,
                    'img2_path': img2_path,
                    'similarity_threshold': config['similarity_threshold'],
                    'max_keypoints': config['max_keypoints']
                })
        
        # Process comparisons in parallel
        duplicates = []
        try:
            # Use multiprocessing to compare images in parallel
            num_processes = min(4, cpu_count())  # Limit to 4 processes to avoid memory issues
            
            with Pool(num_processes) as pool:
                # Process comparisons in batches to avoid overwhelming the system
                batch_size = max(1, len(comparison_tasks) // (num_processes * 2))
                results = []
                
                for i in range(0, len(comparison_tasks), batch_size):
                    if self.is_cancelled:
                        break
                    
                    batch = comparison_tasks[i:i + batch_size]
                    batch_results = pool.map(compare_images_sift_worker, batch)
                    results.extend(batch_results)
                
                # Collect successful matches
                for task, similarity in zip(comparison_tasks, results):
                    if similarity and similarity >= config['similarity_threshold']:
                        duplicates.append({
                            'img1': task['sample_img'],
                            'img2': task['img2'],
                            'similarity': similarity
                        })
                        self.log_updated.emit(f"  📸 Match: {task['sample_img']} ↔ {task['img2']} ({similarity:.1%})")
                        
        except Exception as e:
            self.log_updated.emit(f"Multiprocessing error, falling back to single-threaded: {str(e)}")
            # Fallback to single-threaded comparison
            return self.compare_folders(folder1, folder2, extensions, config)
        
        return duplicates
    
    def get_images_from_folder(self, folder_path, extensions):
        """Get all image files from a folder"""
        try:
            files = os.listdir(folder_path)
            images = []
            for file in files:
                if any(file.lower().endswith(ext.lower()) for ext in extensions):
                    images.append(file)
            return sorted(images)
        except Exception:
            return []
    
    def natural_sort_folders(self, folders):
        """Natural sorting for folder names with decimal support"""
        def extract_sort_key(folder_name):
            # Extract the number part after 'c' (e.g., 'c192.5' -> '192.5')
            match = re.match(r'^c(\d+(?:\.\d+)?)$', folder_name)
            if match:
                number_str = match.group(1)
                if '.' in number_str:
                    # Handle decimal numbers (e.g., 192.5)
                    return float(number_str)
                else:
                    # Handle integer numbers (e.g., 192)
                    return float(number_str)
            
            # Fallback for non-matching patterns
            return float('inf')
        
        return sorted(folders, key=extract_sort_key)
    
    def compare_images_sift(self, img1_path, img2_path):
        """Compare two images using SIFT and return similarity score"""
        try:
            # Load images as grayscale
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                return 0.0
            
            # Extract keypoints and descriptors
            kp1, des1 = self.sift.detectAndCompute(img1, None)
            kp2, des2 = self.sift.detectAndCompute(img2, None)
            
            if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                return 0.0
            
            # FLANN matcher
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            # Calculate similarity score
            if len(kp1) == 0 or len(kp2) == 0:
                return 0.0
                
            similarity = len(good_matches) / max(len(kp1), len(kp2))
            return similarity
            
        except Exception as e:
            return 0.0

class FolderRenamerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.duplicate_worker = None
        self.duplicate_results = []
        self.init_ui()
        self.load_settings()
        
    def init_ui(self):
        self.setWindowTitle("Folder Renamer - Batch Folder Rename Tool")
        self.setGeometry(100, 100, 1000, 700)  # Slightly wider but shorter
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)
        
        # Tab 1: Main settings
        main_tab = QWidget()
        tab_widget.addTab(main_tab, "Main Settings")
        self.setup_main_tab(main_tab)
        
        # Tab 2: Pattern settings
        pattern_tab = QWidget()
        tab_widget.addTab(pattern_tab, "Rename Patterns")
        self.setup_pattern_tab(pattern_tab)
        
        # Tab 3: Preview
        preview_tab = QWidget()
        tab_widget.addTab(preview_tab, "Preview")
        self.setup_preview_tab(preview_tab)
        
        # Tab 4: Duplicate Detection
        duplicate_tab = QWidget()
        tab_widget.addTab(duplicate_tab, "Duplicate Detection")
        self.setup_duplicate_tab(duplicate_tab)
        
        # Control buttons
        self.setup_control_buttons(main_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Log area
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)  # Reduced from 200 to 150
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        main_layout.addWidget(log_group)
        
    def setup_main_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Target directory group
        dir_group = QGroupBox("Target Directory")
        dir_layout = QGridLayout(dir_group)
        
        dir_layout.addWidget(QLabel("Directory to rename folders in:"), 0, 0)
        self.target_dir_edit = QLineEdit()
        self.target_dir_edit.setText(r"h:\manhwa\test_rename")
        dir_layout.addWidget(self.target_dir_edit, 0, 1)
        
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_target_directory)
        dir_layout.addWidget(browse_btn, 0, 2)
        
        layout.addWidget(dir_group)
        
        # Filter group
        filter_group = QGroupBox("Folder Filter")
        filter_layout = QGridLayout(filter_group)
        
        filter_layout.addWidget(QLabel("Filter Pattern:"), 0, 0)
        self.filter_pattern_edit = QLineEdit()
        self.filter_pattern_edit.setPlaceholderText("e.g., c*, chapter*, or leave empty for all folders")
        filter_layout.addWidget(self.filter_pattern_edit, 0, 1)
        
        self.use_regex_checkbox = QCheckBox("Use Regular Expression")
        filter_layout.addWidget(self.use_regex_checkbox, 0, 2)
        
        layout.addWidget(filter_group)
        
        # Sorting group
        sort_group = QGroupBox("Sorting Options")
        sort_layout = QHBoxLayout(sort_group)
        
        self.sort_naturally_checkbox = QCheckBox("Natural sorting (c1, c2, c10 instead of c1, c10, c2)")
        self.sort_naturally_checkbox.setChecked(True)
        sort_layout.addWidget(self.sort_naturally_checkbox)
        
        layout.addWidget(sort_group)
        
        # Mode group
        mode_group = QGroupBox("Operation Mode")
        mode_layout = QHBoxLayout(mode_group)
        
        self.preview_mode_checkbox = QCheckBox("Preview mode (don't actually rename)")
        self.preview_mode_checkbox.setChecked(False)  # Changed to False for actual renaming by default
        mode_layout.addWidget(self.preview_mode_checkbox)
        
        layout.addWidget(mode_group)
        
        layout.addStretch()
        
    def setup_pattern_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Rename method group
        method_group = QGroupBox("Rename Method")
        method_layout = QVBoxLayout(method_group)
        
        # Pattern-based renaming
        self.pattern_radio = QRadioButton("Pattern-based renaming")
        self.pattern_radio.setChecked(True)
        self.pattern_radio.toggled.connect(self.on_rename_method_changed)
        method_layout.addWidget(self.pattern_radio)
        
        pattern_settings = QGroupBox()
        pattern_settings_layout = QGridLayout(pattern_settings)
        
        pattern_settings_layout.addWidget(QLabel("Name Pattern:"), 0, 0)
        self.name_pattern_edit = QLineEdit()
        self.name_pattern_edit.setText("c{chapter}")  # Changed to smart chapter extraction
        self.name_pattern_edit.setPlaceholderText("e.g., c{chapter}, c{ch:03d}, chapter_{index}, {original}_new")
        pattern_settings_layout.addWidget(self.name_pattern_edit, 0, 1)
        
        pattern_settings_layout.addWidget(QLabel("Start Number:"), 1, 0)
        self.start_number_spinbox = QSpinBox()
        self.start_number_spinbox.setRange(0, 9999)
        self.start_number_spinbox.setValue(1)
        pattern_settings_layout.addWidget(self.start_number_spinbox, 1, 1)
        
        method_layout.addWidget(pattern_settings)
        self.pattern_settings_widget = pattern_settings
        
        # Find and replace
        self.replace_radio = QRadioButton("Find and replace")
        self.replace_radio.toggled.connect(self.on_rename_method_changed)
        method_layout.addWidget(self.replace_radio)
        
        replace_settings = QGroupBox()
        replace_settings_layout = QGridLayout(replace_settings)
        
        replace_settings_layout.addWidget(QLabel("Find:"), 0, 0)
        self.find_text_edit = QLineEdit()
        replace_settings_layout.addWidget(self.find_text_edit, 0, 1)
        
        replace_settings_layout.addWidget(QLabel("Replace with:"), 1, 0)
        self.replace_text_edit = QLineEdit()
        replace_settings_layout.addWidget(self.replace_text_edit, 1, 1)
        
        method_layout.addWidget(replace_settings)
        self.replace_settings_widget = replace_settings
        
        # Case conversion
        self.case_radio = QRadioButton("Case conversion")
        self.case_radio.toggled.connect(self.on_rename_method_changed)
        method_layout.addWidget(self.case_radio)
        
        case_settings = QGroupBox()
        case_settings_layout = QHBoxLayout(case_settings)
        
        self.case_combo = QComboBox()
        self.case_combo.addItems(["lower", "upper", "title", "capitalize"])
        case_settings_layout.addWidget(self.case_combo)
        
        method_layout.addWidget(case_settings)
        self.case_settings_widget = case_settings
        
        layout.addWidget(method_group)
        
        # Help text with scroll area
        help_group = QGroupBox("Pattern Help (Scroll to see more)")
        help_group.setMaximumHeight(200)  # Limit height to save space
        help_group.setToolTip("Scroll down to see all available placeholders and examples")
        help_layout = QVBoxLayout(help_group)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setToolTip("Use mouse wheel or scroll bar to see more help content")
        
        # Create help content widget
        help_content = QWidget()
        help_content_layout = QVBoxLayout(help_content)
        
        help_text = QLabel("""
Available placeholders for patterns:

🔢 Sequential Numbers:
• {index} - Sequential number (1, 2, 3...)
• {index:02d} - Zero-padded 2 digits (01, 02, 03...)
• {index:03d} - Zero-padded 3 digits (001, 002, 003...)
• {index:04d} - Zero-padded 4 digits (0001, 0002, 0003...)

📖 Chapter Extraction (Smart):
• {chapter} - Extract chapter number (192.5, 15, etc.)
• {ch} - Same as {chapter} (shorter)
• {chapter:03d} - Zero-padded chapter (192 → 192, 15 → 015)
• {ch:03d} - Same as {chapter:03d}

🔢 Number Extraction:
• {number} - First number found (supports decimals: 192.5)
• {decimal} - Same as {number}
• {integer} - First integer only (192 from 192.5)
• {number2} - Second number found
• {original} - Original folder name

📋 Examples:
• "c{chapter}" → "Chương 192.5" becomes "c192.5"
• "c{ch:03d}" → "Chapter 15" becomes "c015"
• "vol{integer}_ch{decimal}" → "Vol 1 Ch 192.5" becomes "vol1_ch192.5"
• "c{index:03d}" → Sequential: c001, c002, c003...

🎯 Smart Chapter Patterns Detected:
• "Chương 192.5", "Chapter 15", "Ch 20"
• "Tập 10", "192.5 Something", "Something 192.5"

💡 Tips:
• Use {chapter} for automatic chapter detection
• Use {index:03d} for sequential numbering
• Combine placeholders: "vol{integer}_ch{decimal}"
• Test with Preview before actual renaming
        """)
        help_text.setWordWrap(True)
        help_text.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; }")
        help_text.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        help_content_layout.addWidget(help_text)
        help_content_layout.addStretch()
        
        # Set the content widget to scroll area
        scroll_area.setWidget(help_content)
        help_layout.addWidget(scroll_area)
        layout.addWidget(help_group)
        
        self.on_rename_method_changed()  # Initialize visibility
        
        layout.addStretch()
        
    def setup_preview_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Preview controls
        controls_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("Refresh Preview")
        refresh_btn.clicked.connect(self.refresh_preview)
        controls_layout.addWidget(refresh_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Preview list
        preview_group = QGroupBox("Rename Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_list = QListWidget()
        self.preview_list.setFont(QFont("Consolas", 9))
        preview_layout.addWidget(self.preview_list)
        
        layout.addWidget(preview_group)
        
    def setup_duplicate_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Detection settings
        settings_group = QGroupBox("Detection Settings")
        settings_layout = QGridLayout(settings_group)
        
        settings_layout.addWidget(QLabel("Sample Size:"), 0, 0)
        self.sample_size_spinbox = QSpinBox()
        self.sample_size_spinbox.setRange(1, 20)
        self.sample_size_spinbox.setValue(5)
        self.sample_size_spinbox.setToolTip("Number of random images to sample from each folder")
        settings_layout.addWidget(self.sample_size_spinbox, 0, 1)
        
        settings_layout.addWidget(QLabel("Similarity Threshold:"), 0, 2)
        self.similarity_threshold_spinbox = QSpinBox()
        self.similarity_threshold_spinbox.setRange(10, 95)
        self.similarity_threshold_spinbox.setValue(70)
        self.similarity_threshold_spinbox.setSuffix("%")
        self.similarity_threshold_spinbox.setToolTip("Minimum similarity percentage to consider as duplicate")
        settings_layout.addWidget(self.similarity_threshold_spinbox, 0, 3)
        
        settings_layout.addWidget(QLabel("Max Keypoints:"), 1, 0)
        self.max_keypoints_spinbox = QSpinBox()
        self.max_keypoints_spinbox.setRange(100, 5000)
        self.max_keypoints_spinbox.setValue(1000)
        self.max_keypoints_spinbox.setToolTip("Maximum SIFT keypoints to extract per image")
        settings_layout.addWidget(self.max_keypoints_spinbox, 1, 1)
        
        settings_layout.addWidget(QLabel("Min Images per Folder:"), 1, 2)
        self.min_images_spinbox = QSpinBox()
        self.min_images_spinbox.setRange(1, 100)
        self.min_images_spinbox.setValue(5)
        self.min_images_spinbox.setToolTip("Skip folders with fewer images than this")
        settings_layout.addWidget(self.min_images_spinbox, 1, 3)
        
        settings_layout.addWidget(QLabel("Image Extensions:"), 2, 0)
        self.image_extensions_edit = QLineEdit()
        self.image_extensions_edit.setText(".jpg, .jpeg, .png, .webp")
        self.image_extensions_edit.setToolTip("Comma-separated list of image extensions")
        settings_layout.addWidget(self.image_extensions_edit, 2, 1, 1, 3)
        
        # Performance settings
        settings_layout.addWidget(QLabel("Enable Multiprocessing:"), 3, 0)
        self.enable_multiprocessing_checkbox = QCheckBox("Use parallel processing for faster comparison")
        self.enable_multiprocessing_checkbox.setChecked(True)
        self.enable_multiprocessing_checkbox.setToolTip("Use multiple CPU cores to speed up image comparison")
        settings_layout.addWidget(self.enable_multiprocessing_checkbox, 3, 1, 1, 2)
        
        settings_layout.addWidget(QLabel("Compare Mode:"), 3, 3)
        self.compare_mode_combo = QComboBox()
        self.compare_mode_combo.addItems(["Adjacent only (c1↔c2, c2↔c3)", "All pairs (c1↔c2, c1↔c3, c2↔c3)"])
        self.compare_mode_combo.setCurrentIndex(0)  # Default to adjacent only
        self.compare_mode_combo.setToolTip("Choose comparison strategy: adjacent folders or all combinations")
        settings_layout.addWidget(self.compare_mode_combo, 4, 0, 1, 4)
        
        layout.addWidget(settings_group)
        
        # Detection controls
        controls_layout = QHBoxLayout()
        
        self.detect_btn = QPushButton("Start Duplicate Detection")
        self.detect_btn.clicked.connect(self.start_duplicate_detection)
        self.detect_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        controls_layout.addWidget(self.detect_btn)
        
        self.cancel_detect_btn = QPushButton("Cancel Detection")
        self.cancel_detect_btn.clicked.connect(self.cancel_duplicate_detection)
        self.cancel_detect_btn.setEnabled(False)
        self.cancel_detect_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
        controls_layout.addWidget(self.cancel_detect_btn)
        
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self.export_duplicate_results)
        controls_layout.addWidget(export_btn)
        
        clear_btn = QPushButton("Clear Results")
        clear_btn.clicked.connect(self.clear_duplicate_results)
        controls_layout.addWidget(clear_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Progress bar for detection
        self.detection_progress_bar = QProgressBar()
        self.detection_progress_bar.setVisible(False)
        layout.addWidget(self.detection_progress_bar)
        
        # Results display
        results_group = QGroupBox("Duplicate Detection Results")
        results_layout = QVBoxLayout(results_group)
        
        # Results tree
        self.duplicate_tree = QTreeWidget()
        self.duplicate_tree.setHeaderLabels(["Folder Pair / Duplicate", "Similarity", "Actions"])
        self.duplicate_tree.setFont(QFont("Consolas", 9))
        results_layout.addWidget(self.duplicate_tree)
        
        layout.addWidget(results_group)
        
        layout.addStretch()
        
    def setup_control_buttons(self, layout):
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Renaming")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        button_layout.addWidget(self.start_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
        button_layout.addWidget(self.cancel_btn)
        
        save_settings_btn = QPushButton("Save Settings")
        save_settings_btn.clicked.connect(self.save_settings)
        button_layout.addWidget(save_settings_btn)
        
        load_settings_btn = QPushButton("Load Settings")
        load_settings_btn.clicked.connect(self.load_settings)
        button_layout.addWidget(load_settings_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
    def browse_target_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Target Directory")
        if directory:
            self.target_dir_edit.setText(directory)
            self.refresh_preview()
            
    def on_rename_method_changed(self):
        """Enable/disable controls based on rename method"""
        is_pattern = self.pattern_radio.isChecked()
        is_replace = self.replace_radio.isChecked()
        is_case = self.case_radio.isChecked()
        
        self.pattern_settings_widget.setEnabled(is_pattern)
        self.replace_settings_widget.setEnabled(is_replace)
        self.case_settings_widget.setEnabled(is_case)
        
    def get_duplicate_config(self):
        """Get configuration for duplicate detection"""
        return {
            'target_directory': self.target_dir_edit.text(),
            'sample_size': self.sample_size_spinbox.value(),
            'similarity_threshold': self.similarity_threshold_spinbox.value() / 100.0,
            'max_keypoints': self.max_keypoints_spinbox.value(),
            'min_images_per_folder': self.min_images_spinbox.value(),
            'image_extensions': self.image_extensions_edit.text(),
            'enable_multiprocessing': self.enable_multiprocessing_checkbox.isChecked(),
            'compare_adjacent_only': self.compare_mode_combo.currentIndex() == 0
        }
        
    def get_config(self):
        rename_method = 'pattern' if self.pattern_radio.isChecked() else \
                      'replace' if self.replace_radio.isChecked() else 'case'
        
        return {
            'target_directory': self.target_dir_edit.text(),
            'filter_pattern': self.filter_pattern_edit.text(),
            'use_regex': self.use_regex_checkbox.isChecked(),
            'sort_naturally': self.sort_naturally_checkbox.isChecked(),
            'preview_mode': self.preview_mode_checkbox.isChecked(),
            'rename_method': rename_method,
            'name_pattern': self.name_pattern_edit.text(),
            'start_number': self.start_number_spinbox.value(),
            'find_text': self.find_text_edit.text(),
            'replace_text': self.replace_text_edit.text(),
            'case_option': self.case_combo.currentText()
        }
        
    def refresh_preview(self):
        """Refresh preview list"""
        config = self.get_config()
        
        if not os.path.exists(config['target_directory']):
            self.preview_list.clear()
            return
        
        try:
            items = os.listdir(config['target_directory'])
            folders = [item for item in items if os.path.isdir(os.path.join(config['target_directory'], item))]
            
            if config['filter_pattern']:
                # Apply filter
                pattern = config['filter_pattern']
                if config['use_regex']:
                    try:
                        regex = re.compile(pattern)
                        folders = [f for f in folders if regex.search(f)]
                    except re.error:
                        pass
                else:
                    pattern = pattern.replace('*', '.*').replace('?', '.')
                    try:
                        regex = re.compile(pattern)
                        folders = [f for f in folders if regex.search(f)]
                    except re.error:
                        folders = [f for f in folders if pattern in f]
            
            # Sort folders
            if config['sort_naturally']:
                # Enhanced natural sorting with decimal support
                def natural_key(text):
                    parts = []
                    for part in re.split(r'(\d+(?:\.\d+)?)', text):
                        if re.match(r'\d+(?:\.\d+)?$', part):
                            # Convert to float for proper decimal sorting
                            parts.append(float(part))
                        else:
                            # Keep text parts as lowercase
                            parts.append(part.lower())
                    return parts
                folders = sorted(folders, key=natural_key)
            else:
                folders = sorted(folders)
            
            # Generate preview
            self.preview_list.clear()
            
            worker = FolderRenamerWorker(config)
            for i, folder_name in enumerate(folders):
                new_name = worker.generate_new_name(folder_name, i, config)
                
                item = QListWidgetItem()
                if folder_name == new_name:
                    item.setText(f"{folder_name} (no change)")
                    item.setForeground(Qt.GlobalColor.gray)
                else:
                    item.setText(f"{folder_name} → {new_name}")
                    item.setForeground(Qt.GlobalColor.blue)
                
                self.preview_list.addItem(item)
                
        except Exception as e:
            QMessageBox.warning(self, "Preview Error", f"Error generating preview: {str(e)}")
        
    def start_processing(self):
        config = self.get_config()
        
        # Validation
        if not config['target_directory']:
            QMessageBox.warning(self, "Warning", "Please select target directory!")
            return
        
        if not os.path.exists(config['target_directory']):
            QMessageBox.warning(self, "Warning", "Target directory does not exist!")
            return
        
        # Clear log
        self.log_text.clear()
        
        # Setup UI for processing
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start worker thread
        self.worker = FolderRenamerWorker(config)
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.log_updated.connect(self.add_log)
        self.worker.finished.connect(self.processing_finished)
        self.worker.start()
        
    def cancel_processing(self):
        if self.worker:
            self.worker.cancel()
            
    def processing_finished(self, success, message):
        # Reset UI
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        # Refresh preview after renaming
        self.refresh_preview()
        
        # Show completion message
        if success:
            self.add_log(f"\n✅ {message}")
            QMessageBox.information(self, "Success", message)
        else:
            self.add_log(f"\n❌ {message}")
            QMessageBox.critical(self, "Error", message)
            
    def add_log(self, message):
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        
    def save_settings(self):
        config = self.get_config()
        # Add duplicate detection settings
        config.update({
            'sample_size': self.sample_size_spinbox.value(),
            'similarity_threshold_percent': self.similarity_threshold_spinbox.value(),
            'max_keypoints': self.max_keypoints_spinbox.value(),
            'min_images_per_folder': self.min_images_spinbox.value(),
            'image_extensions': self.image_extensions_edit.text(),
            'enable_multiprocessing': self.enable_multiprocessing_checkbox.isChecked(),
            'compare_adjacent_only': self.compare_mode_combo.currentIndex() == 0
        })
        
        try:
            with open("folder_renamer_settings.json", "w") as f:
                json.dump(config, f, indent=2)
            QMessageBox.information(self, "Success", "Settings saved successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {str(e)}")
            
    def load_settings(self):
        try:
            if os.path.exists("folder_renamer_settings.json"):
                with open("folder_renamer_settings.json", "r") as f:
                    config = json.load(f)
                    
                # Load settings into UI
                self.target_dir_edit.setText(config.get('target_directory', ''))
                self.filter_pattern_edit.setText(config.get('filter_pattern', ''))
                self.use_regex_checkbox.setChecked(config.get('use_regex', False))
                self.sort_naturally_checkbox.setChecked(config.get('sort_naturally', True))
                self.preview_mode_checkbox.setChecked(config.get('preview_mode', False))
                
                # Rename method
                method = config.get('rename_method', 'pattern')
                if method == 'pattern':
                    self.pattern_radio.setChecked(True)
                elif method == 'replace':
                    self.replace_radio.setChecked(True)
                elif method == 'case':
                    self.case_radio.setChecked(True)
                
                self.name_pattern_edit.setText(config.get('name_pattern', 'c{chapter}'))  # Updated default
                self.start_number_spinbox.setValue(config.get('start_number', 1))
                self.find_text_edit.setText(config.get('find_text', ''))
                self.replace_text_edit.setText(config.get('replace_text', ''))
                self.case_combo.setCurrentText(config.get('case_option', 'lower'))
                
                # Load duplicate detection settings
                self.sample_size_spinbox.setValue(config.get('sample_size', 5))
                self.similarity_threshold_spinbox.setValue(config.get('similarity_threshold_percent', 70))
                self.max_keypoints_spinbox.setValue(config.get('max_keypoints', 1000))
                self.min_images_spinbox.setValue(config.get('min_images_per_folder', 5))
                self.image_extensions_edit.setText(config.get('image_extensions', '.jpg, .jpeg, .png, .webp'))
                self.enable_multiprocessing_checkbox.setChecked(config.get('enable_multiprocessing', True))
                self.compare_mode_combo.setCurrentIndex(0 if config.get('compare_adjacent_only', True) else 1)
                
                self.on_rename_method_changed()
                
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to load settings: {str(e)}")
    
    def start_duplicate_detection(self):
        """Start duplicate detection process"""
        config = self.get_duplicate_config()
        
        # Validation
        if not config['target_directory']:
            QMessageBox.warning(self, "Warning", "Please select target directory!")
            return
        
        if not os.path.exists(config['target_directory']):
            QMessageBox.warning(self, "Warning", "Target directory does not exist!")
            return
        
        # Clear previous results
        self.duplicate_results = []
        self.duplicate_tree.clear()
        
        # Setup UI for detection
        self.detect_btn.setEnabled(False)
        self.cancel_detect_btn.setEnabled(True)
        self.detection_progress_bar.setVisible(True)
        self.detection_progress_bar.setValue(0)
        
        # Start worker thread
        self.duplicate_worker = DuplicateDetectionWorker(config)
        self.duplicate_worker.progress_updated.connect(self.detection_progress_bar.setValue)
        self.duplicate_worker.log_updated.connect(self.add_log)
        self.duplicate_worker.finished.connect(self.duplicate_detection_finished)
        self.duplicate_worker.duplicate_found.connect(self.add_duplicate_result)
        self.duplicate_worker.start()
        
    def cancel_duplicate_detection(self):
        """Cancel duplicate detection process"""
        if self.duplicate_worker:
            self.duplicate_worker.cancel()
            
    def duplicate_detection_finished(self, success, message):
        """Handle duplicate detection completion"""
        # Reset UI
        self.detect_btn.setEnabled(True)
        self.cancel_detect_btn.setEnabled(False)
        self.detection_progress_bar.setVisible(False)
        
        # Show completion message
        if success:
            self.add_log(f"\n✅ {message}")
            QMessageBox.information(self, "Detection Complete", message)
        else:
            self.add_log(f"\n❌ {message}")
            QMessageBox.critical(self, "Detection Error", message)
    
    def add_duplicate_result(self, folder1, img1, folder2, img2, similarity):
        """Add a duplicate result to the tree"""
        # Find or create folder pair item
        pair_name = f"{folder1} ↔ {folder2}"
        pair_item = None
        
        # Search for existing pair item
        for i in range(self.duplicate_tree.topLevelItemCount()):
            item = self.duplicate_tree.topLevelItem(i)
            if item.text(0) == pair_name:
                pair_item = item
                break
        
        # Create new pair item if not found
        if pair_item is None:
            pair_item = QTreeWidgetItem(self.duplicate_tree)
            pair_item.setText(0, pair_name)
            pair_item.setText(1, "")
            pair_item.setText(2, "")
            pair_item.setExpanded(True)
        
        # Add duplicate item
        duplicate_item = QTreeWidgetItem(pair_item)
        duplicate_item.setText(0, f"{img1} ↔ {img2}")
        duplicate_item.setText(1, f"{similarity:.1%}")
        duplicate_item.setText(2, "📸 View")
        
        # Store result data
        self.duplicate_results.append({
            'folder1': folder1,
            'img1': img1,
            'folder2': folder2,
            'img2': img2,
            'similarity': similarity
        })
        
        # Update pair item count
        count = pair_item.childCount()
        pair_item.setText(1, f"{count} duplicates")
    
    def export_duplicate_results(self):
        """Export duplicate results to text file"""
        if not self.duplicate_results:
            QMessageBox.information(self, "No Results", "No duplicate results to export!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Duplicate Report", "duplicate_report.txt", "Text Files (*.txt)"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("🔍 DUPLICATE DETECTION REPORT\n")
                    f.write("═" * 50 + "\n\n")
                    
                    # Group by folder pairs
                    pairs = {}
                    for result in self.duplicate_results:
                        pair_key = f"{result['folder1']} ↔ {result['folder2']}"
                        if pair_key not in pairs:
                            pairs[pair_key] = []
                        pairs[pair_key].append(result)
                    
                    for pair_key, duplicates in pairs.items():
                        f.write(f"📁 {pair_key}:\n")
                        f.write(f"   ✅ Found {len(duplicates)} potential duplicates:\n")
                        for dup in duplicates:
                            f.write(f"   • {dup['img1']} ↔ {dup['img2']} (Similarity: {dup['similarity']:.1%})\n")
                        f.write("\n")
                    
                    f.write(f"🎯 SUMMARY:\n")
                    f.write(f"   • Total duplicates found: {len(self.duplicate_results)}\n")
                    f.write(f"   • Folder pairs with duplicates: {len(pairs)}\n")
                
                QMessageBox.information(self, "Export Complete", f"Report saved to {filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to save report: {str(e)}")
    
    def clear_duplicate_results(self):
        """Clear all duplicate results"""
        self.duplicate_results = []
        self.duplicate_tree.clear()
        self.add_log("Duplicate results cleared")

def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Folder Renamer")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Manga Tools")
    
    # Create and show main window
    window = FolderRenamerGUI()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()