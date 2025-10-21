import sys
import os
import json
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
                             QProgressBar, QTextEdit, QFileDialog, QGroupBox,
                             QGridLayout, QMessageBox, QTabWidget)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QIcon
import threading
import time
from PIL import Image
import math
from multiprocessing import Pool, cpu_count
from functools import partial

# Import từ module gốc
from image_splitter import (split_image_large, process_images_batch, 
                           create_batches, get_image_info, estimate_file_size_kb,
                           find_optimal_segment_height)

Image.MAX_IMAGE_PIXELS = None

class ImageSplitterWorker(QThread):
    """Worker thread để xử lý ảnh không block UI"""
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
            self.process_images()
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")
    
    def process_images(self):
        config = self.config
        
        # Validation
        if not os.path.exists(config['input_directory']):
            self.finished.emit(False, "Input directory not found!")
            return
        
        if not os.path.exists(config['output_directory']):
            os.makedirs(config['output_directory'], exist_ok=True)
        
        # Find image files
        try:
            files = os.listdir(config['input_directory'])
        except Exception as e:
            self.finished.emit(False, f"Error reading directory: {str(e)}")
            return
        
        image_files = [f for f in files if f.startswith(config['file_prefix']) 
                      and f.endswith(config['file_extension'])]
        
        if not image_files:
            self.finished.emit(False, "No image files found!")
            return
        
        self.log_updated.emit(f"Found {len(image_files)} images to process")
        
        # Prepare image info
        image_info_list = []
        for i, image_file in enumerate(sorted(image_files)):
            if self.is_cancelled:
                return
                
            image_path = os.path.join(config['input_directory'], image_file)
            folder_name = os.path.splitext(image_file)[0]
            output_dir = os.path.join(config['output_directory'], folder_name)
            image_info_list.append((image_path, output_dir, image_file))
            
            # Update progress
            progress = int((i + 1) / len(image_files) * 10)  # 10% for preparation
            self.progress_updated.emit(progress)
        
        # Process images
        output_config = (config['output_format'], config['output_extension'], 
                        config['jpeg_quality'], config['jpeg_optimize'], 
                        config['jpeg_progressive'])
        
        # Update global variables for the worker functions
        import image_splitter
        image_splitter.ENABLE_SIZE_BASED_SPLITTING = config['enable_size_based']
        image_splitter.TARGET_FILE_SIZE_KB = config['target_file_size_kb']
        image_splitter.MAX_FILE_SIZE_KB = config['max_file_size_kb']
        image_splitter.MIN_SEGMENT_HEIGHT = config['min_segment_height']
        image_splitter.SIZE_TOLERANCE = config['size_tolerance']
        
        batches = list(create_batches(image_info_list, config['batch_size']))
        total_processed = 0
        success_count = 0
        error_count = 0
        
        for batch_idx, batch in enumerate(batches):
            if self.is_cancelled:
                return
            
            # Process batch
            results = process_images_batch(batch, config['segment_height'], 
                                         config['show_detailed_log'], output_config)
            
            # Update results
            for result in results:
                total_processed += 1
                if result.startswith("✓"):
                    success_count += 1
                else:
                    error_count += 1
                
                self.log_updated.emit(result)
                
                # Update progress (10% prep + 90% processing)
                progress = 10 + int((total_processed / len(image_info_list)) * 90)
                self.progress_updated.emit(progress)
        
        # Finish
        summary = f"Completed! Success: {success_count}/{len(image_files)}, Errors: {error_count}"
        self.finished.emit(True, summary)

class ImageSplitterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()
        self.load_settings()
        
    def init_ui(self):
        self.setWindowTitle("Image Splitter - Manga Processing Tool")
        self.setGeometry(100, 100, 800, 900)
        
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
        
        # Tab 2: Advanced settings
        advanced_tab = QWidget()
        tab_widget.addTab(advanced_tab, "Advanced Settings")
        self.setup_advanced_tab(advanced_tab)
        
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
        self.log_text.setMaximumHeight(200)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        main_layout.addWidget(log_group)
        
    def setup_main_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Paths group
        paths_group = QGroupBox("File Paths")
        paths_layout = QGridLayout(paths_group)
        
        # Input directory
        paths_layout.addWidget(QLabel("Input Directory:"), 0, 0)
        self.input_dir_edit = QLineEdit()
        self.input_dir_edit.setText(r"h:\manhwa\The_Martial_God_Who_Regressed_Back_to_Level_2\result")
        paths_layout.addWidget(self.input_dir_edit, 0, 1)
        
        input_browse_btn = QPushButton("Browse")
        input_browse_btn.clicked.connect(self.browse_input_directory)
        paths_layout.addWidget(input_browse_btn, 0, 2)
        
        # Output directory
        paths_layout.addWidget(QLabel("Output Directory:"), 1, 0)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText(r"h:\manhwa\The_Martial_God_Who_Regressed_Back_to_Level_2\finish")
        paths_layout.addWidget(self.output_dir_edit, 1, 1)
        
        output_browse_btn = QPushButton("Browse")
        output_browse_btn.clicked.connect(self.browse_output_directory)
        paths_layout.addWidget(output_browse_btn, 1, 2)
        
        layout.addWidget(paths_group)
        
        # File settings group
        file_group = QGroupBox("File Settings")
        file_layout = QGridLayout(file_group)
        
        file_layout.addWidget(QLabel("File Prefix:"), 0, 0)
        self.file_prefix_edit = QLineEdit("c")
        file_layout.addWidget(self.file_prefix_edit, 0, 1)
        
        file_layout.addWidget(QLabel("Input Extension:"), 0, 2)
        self.input_ext_combo = QComboBox()
        self.input_ext_combo.addItems([".png", ".jpg", ".jpeg", ".bmp"])
        file_layout.addWidget(self.input_ext_combo, 0, 3)
        
        file_layout.addWidget(QLabel("Output Format:"), 1, 0)
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["JPEG", "PNG"])
        self.output_format_combo.currentTextChanged.connect(self.on_output_format_changed)
        file_layout.addWidget(self.output_format_combo, 1, 1)
        
        file_layout.addWidget(QLabel("Output Extension:"), 1, 2)
        self.output_ext_edit = QLineEdit(".jpg")
        file_layout.addWidget(self.output_ext_edit, 1, 3)
        
        layout.addWidget(file_group)
        
        # Splitting method group
        split_group = QGroupBox("Splitting Method")
        split_layout = QVBoxLayout(split_group)
        
        # Size-based splitting
        self.size_based_checkbox = QCheckBox("Enable Size-based Splitting")
        self.size_based_checkbox.setChecked(True)
        self.size_based_checkbox.toggled.connect(self.on_split_method_changed)
        split_layout.addWidget(self.size_based_checkbox)
        
        # Size settings
        size_settings_layout = QGridLayout()
        
        size_settings_layout.addWidget(QLabel("Target File Size (KB):"), 0, 0)
        self.target_size_spinbox = QSpinBox()
        self.target_size_spinbox.setRange(50, 2000)
        self.target_size_spinbox.setValue(300)
        size_settings_layout.addWidget(self.target_size_spinbox, 0, 1)
        
        size_settings_layout.addWidget(QLabel("Max File Size (KB):"), 0, 2)
        self.max_size_spinbox = QSpinBox()
        self.max_size_spinbox.setRange(100, 5000)
        self.max_size_spinbox.setValue(500)
        size_settings_layout.addWidget(self.max_size_spinbox, 0, 3)
        
        size_settings_layout.addWidget(QLabel("Min Segment Height (px):"), 1, 0)
        self.min_height_spinbox = QSpinBox()
        self.min_height_spinbox.setRange(500, 5000)
        self.min_height_spinbox.setValue(1000)
        size_settings_layout.addWidget(self.min_height_spinbox, 1, 1)
        
        size_settings_layout.addWidget(QLabel("Size Tolerance:"), 1, 2)
        self.tolerance_spinbox = QDoubleSpinBox()
        self.tolerance_spinbox.setRange(0.05, 0.5)
        self.tolerance_spinbox.setSingleStep(0.05)
        self.tolerance_spinbox.setValue(0.1)
        size_settings_layout.addWidget(self.tolerance_spinbox, 1, 3)
        
        split_layout.addLayout(size_settings_layout)
        
        # Height-based splitting
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("Segment Height (px):"))
        self.segment_height_spinbox = QSpinBox()
        self.segment_height_spinbox.setRange(1000, 20000)
        self.segment_height_spinbox.setValue(8000)
        height_layout.addWidget(self.segment_height_spinbox)
        height_layout.addStretch()
        
        split_layout.addLayout(height_layout)
        layout.addWidget(split_group)
        
        layout.addStretch()
        
    def setup_advanced_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Processing settings
        processing_group = QGroupBox("Processing Settings")
        processing_layout = QGridLayout(processing_group)
        
        processing_layout.addWidget(QLabel("Number of Processes:"), 0, 0)
        self.num_processes_spinbox = QSpinBox()
        self.num_processes_spinbox.setRange(1, cpu_count())
        self.num_processes_spinbox.setValue(min(20, cpu_count()))
        processing_layout.addWidget(self.num_processes_spinbox, 0, 1)
        
        processing_layout.addWidget(QLabel("Batch Size:"), 0, 2)
        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setRange(1, 10)
        self.batch_size_spinbox.setValue(2)
        processing_layout.addWidget(self.batch_size_spinbox, 0, 3)
        
        layout.addWidget(processing_group)
        
        # JPEG settings
        jpeg_group = QGroupBox("JPEG Quality Settings")
        jpeg_layout = QGridLayout(jpeg_group)
        
        jpeg_layout.addWidget(QLabel("JPEG Quality:"), 0, 0)
        self.jpeg_quality_spinbox = QSpinBox()
        self.jpeg_quality_spinbox.setRange(60, 100)
        self.jpeg_quality_spinbox.setValue(95)
        jpeg_layout.addWidget(self.jpeg_quality_spinbox, 0, 1)
        
        self.jpeg_optimize_checkbox = QCheckBox("Optimize JPEG")
        self.jpeg_optimize_checkbox.setChecked(True)
        jpeg_layout.addWidget(self.jpeg_optimize_checkbox, 0, 2)
        
        self.jpeg_progressive_checkbox = QCheckBox("Progressive JPEG")
        self.jpeg_progressive_checkbox.setChecked(False)
        jpeg_layout.addWidget(self.jpeg_progressive_checkbox, 0, 3)
        
        layout.addWidget(jpeg_group)
        
        # Logging settings
        log_group = QGroupBox("Logging Settings")
        log_layout = QVBoxLayout(log_group)
        
        self.detailed_log_checkbox = QCheckBox("Show Detailed Log")
        self.detailed_log_checkbox.setChecked(False)
        log_layout.addWidget(self.detailed_log_checkbox)
        
        layout.addWidget(log_group)
        
        layout.addStretch()
        
    def setup_control_buttons(self, layout):
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Processing")
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
        
    def browse_input_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.input_dir_edit.setText(directory)
            
    def browse_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_edit.setText(directory)
            
    def on_output_format_changed(self, format_type):
        if format_type == "JPEG":
            self.output_ext_edit.setText(".jpg")
        elif format_type == "PNG":
            self.output_ext_edit.setText(".png")
            
    def on_split_method_changed(self, checked):
        # Enable/disable controls based on splitting method
        pass
        
    def get_config(self):
        return {
            'input_directory': self.input_dir_edit.text(),
            'output_directory': self.output_dir_edit.text(),
            'file_prefix': self.file_prefix_edit.text(),
            'file_extension': self.input_ext_combo.currentText(),
            'output_format': self.output_format_combo.currentText(),
            'output_extension': self.output_ext_edit.text(),
            'enable_size_based': self.size_based_checkbox.isChecked(),
            'target_file_size_kb': self.target_size_spinbox.value(),
            'max_file_size_kb': self.max_size_spinbox.value(),
            'min_segment_height': self.min_height_spinbox.value(),
            'size_tolerance': self.tolerance_spinbox.value(),
            'segment_height': self.segment_height_spinbox.value(),
            'num_processes': self.num_processes_spinbox.value(),
            'batch_size': self.batch_size_spinbox.value(),
            'jpeg_quality': self.jpeg_quality_spinbox.value(),
            'jpeg_optimize': self.jpeg_optimize_checkbox.isChecked(),
            'jpeg_progressive': self.jpeg_progressive_checkbox.isChecked(),
            'show_detailed_log': self.detailed_log_checkbox.isChecked()
        }
        
    def start_processing(self):
        config = self.get_config()
        
        # Validation
        if not config['input_directory'] or not config['output_directory']:
            QMessageBox.warning(self, "Warning", "Please select input and output directories!")
            return
            
        # Clear log
        self.log_text.clear()
        
        # Setup UI for processing
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start worker thread
        self.worker = ImageSplitterWorker(config)
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
        try:
            with open("image_splitter_settings.json", "w") as f:
                json.dump(config, f, indent=2)
            QMessageBox.information(self, "Success", "Settings saved successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {str(e)}")
            
    def load_settings(self):
        try:
            if os.path.exists("image_splitter_settings.json"):
                with open("image_splitter_settings.json", "r") as f:
                    config = json.load(f)
                    
                # Load settings into UI
                self.input_dir_edit.setText(config.get('input_directory', ''))
                self.output_dir_edit.setText(config.get('output_directory', ''))
                self.file_prefix_edit.setText(config.get('file_prefix', 'c'))
                self.input_ext_combo.setCurrentText(config.get('file_extension', '.png'))
                self.output_format_combo.setCurrentText(config.get('output_format', 'JPEG'))
                self.output_ext_edit.setText(config.get('output_extension', '.jpg'))
                self.size_based_checkbox.setChecked(config.get('enable_size_based', True))
                self.target_size_spinbox.setValue(config.get('target_file_size_kb', 300))
                self.max_size_spinbox.setValue(config.get('max_file_size_kb', 500))
                self.min_height_spinbox.setValue(config.get('min_segment_height', 1000))
                self.tolerance_spinbox.setValue(config.get('size_tolerance', 0.1))
                self.segment_height_spinbox.setValue(config.get('segment_height', 8000))
                self.num_processes_spinbox.setValue(config.get('num_processes', min(20, cpu_count())))
                self.batch_size_spinbox.setValue(config.get('batch_size', 2))
                self.jpeg_quality_spinbox.setValue(config.get('jpeg_quality', 95))
                self.jpeg_optimize_checkbox.setChecked(config.get('jpeg_optimize', True))
                self.jpeg_progressive_checkbox.setChecked(config.get('jpeg_progressive', False))
                self.detailed_log_checkbox.setChecked(config.get('show_detailed_log', False))
                
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to load settings: {str(e)}")

def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Image Splitter")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Manga Tools")
    
    # Create and show main window
    window = ImageSplitterGUI()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()