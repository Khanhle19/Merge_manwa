import os
import re
import glob
import shutil
from PIL import Image
from tqdm import tqdm

# ===== LOGIC CHUYỂN ĐỔI ĐỊNH DẠNG ẢNH =====
def convert_images_in_all_subfolders(root_dir):
    """Chuyển đổi tất cả ảnh trong các subfolder thành định dạng JPG"""
    print(f"Converting images to JPG format in: {root_dir}")
    
    # Các định dạng ảnh phổ biến
    valid_exts = ('.png', '.bmp', '.jpeg', '.webp', '.tiff', '.gif', '.jpg')
    converted_count = 0
    error_count = 0
    
    # Duyệt qua tất cả các thư mục con trong root_dir
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
            
        print(f"Processing subfolder: {subfolder}")

        # Duyệt qua tất cả file ảnh trong thư mục con
        for filename in os.listdir(subfolder_path):
            if filename.lower().endswith(valid_exts):
                file_path = os.path.join(subfolder_path, filename)
                try:
                    # Đọc ảnh
                    img = Image.open(file_path)
                    
                    # Đặt tên file .jpg mới (giữ nguyên tên, chỉ đổi đuôi)
                    new_filename = os.path.splitext(filename)[0] + ".jpg"
                    new_file_path = os.path.join(subfolder_path, new_filename)
                    
                    # Nếu file đã là .jpg thì bỏ qua
                    if filename.lower().endswith('.jpg'):
                        continue 
                    
                    # Lưu ảnh dưới dạng JPG với chất lượng cao
                    img = img.convert("RGB")
                    img.save(new_file_path, "JPEG", quality=95)
                    print(f"✅ Converted: {filename} -> {new_filename}")
                    
                    # Xóa file cũ
                    os.remove(file_path)
                    converted_count += 1
                    
                except Exception as e:
                    print(f"❌ Error converting {file_path}: {e}")
                    error_count += 1
    
    print(f"Image conversion completed! Converted: {converted_count}, Errors: {error_count}")

# ===== LOGIC ĐỔI TÊN FOLDER CHAP =====
def rename_chap_folders(base_dir):
    """Đổi tên folder từ các format phức tạp thành 'cX' (hỗ trợ Volume, Chapter, số thập phân)"""
    print(f"🔧 IMPROVED VERSION: Renaming chapter folders in: {base_dir}")
    print("📋 Examples: 'cCh.293' -> 'c293', 'cVol.02Ch.001.5' -> 'c1.5', 'cVol.TBDCh.044.5' -> 'c44.5'")
    
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    renamed_count = 0
    
    def extract_chapter_number(folder_name):
        """Extract chapter number from complex folder names"""
        
        # Pattern 1: cVol.XXCh.YYY.Z format
        # Examples: cVol.02Ch.001.5 -> 1.5, cVol.3Ch.20 -> 20, cVol.21Ch.192.1 -> 192.1
        vol_ch_match = re.search(r"c?Vol\.(?:\d+|TBD)Ch\.(\d+(?:\.\d+)?)", folder_name, re.IGNORECASE)
        if vol_ch_match:
            chap_num = vol_ch_match.group(1)
            # Remove leading zeros but keep decimal part
            if '.' in chap_num:
                main_part, decimal_part = chap_num.split('.')
                chap_num = str(int(main_part)) + '.' + decimal_part
            else:
                chap_num = str(int(chap_num))
            return chap_num
        
        # Pattern 2: cCh.XXX format
        # Examples: cCh.293 -> 293, cCh.001 -> 1
        ch_match = re.search(r"c?Ch\.(\d+(?:\.\d+)?)", folder_name, re.IGNORECASE)
        if ch_match:
            chap_num = ch_match.group(1)
            # Remove leading zeros but keep decimal part
            if '.' in chap_num:
                main_part, decimal_part = chap_num.split('.')
                chap_num = str(int(main_part)) + '.' + decimal_part
            else:
                chap_num = str(int(chap_num))
            return chap_num
        
        # Pattern 3: Standard "Chap" + số
        # Examples: "Chap 85", "Chapter 12", "cChap85"
        chap_match = re.search(r"c?Chap(?:ter)?\s*(\d+(?:\.\d+)?)", folder_name, re.IGNORECASE)
        if chap_match:
            return chap_match.group(1)
        
        # Pattern 4: Volume + Chapter format (other variations)
        # Examples: "Vol 2 Ch 15", "Volume 1 Chapter 5"
        vol_chapter_match = re.search(r"Vol(?:ume)?\s*\d+\s*Ch(?:apter)?\s*(\d+(?:\.\d+)?)", folder_name, re.IGNORECASE)
        if vol_chapter_match:
            return vol_chapter_match.group(1)
        
        # Pattern 5: Vietnamese format
        # Examples: "Chương 5", "chương 10"
        vn_match = re.search(r"c?Ch[ươuư][ơương]ng\s*(\d+(?:\.\d+)?)", folder_name, re.IGNORECASE)
        if vn_match:
            return vn_match.group(1)
        
        # Pattern 6: Korean format
        # Examples: "제 3화", "c제5화"
        kr_match = re.search(r"c?제\s*(\d+(?:\.\d+)?)\s*화?", folder_name, re.IGNORECASE)
        if kr_match:
            return kr_match.group(1)
        
        # Pattern 7: Already in correct format
        # Examples: "c123", "c45.5"
        correct_match = re.match(r"^c(\d+(?:\.\d+)?)$", folder_name, re.IGNORECASE)
        if correct_match:
            return "ALREADY_CORRECT"
        
        return None
    
    with tqdm(folders, desc="Renaming chapters", unit="folder") as pbar:
        for folder_name in pbar:
            old_path = os.path.join(base_dir, folder_name)
            pbar.set_postfix_str(f"Processing {folder_name}")
            
            # Extract chapter number
            chap_num = extract_chapter_number(folder_name)
            
            if chap_num == "ALREADY_CORRECT":
                pbar.set_postfix_str(f"Already correct: {folder_name}")
                continue
            elif chap_num:
                new_name = f"c{chap_num}"
                
                # Only rename if different from current name
                if new_name != folder_name:
                    new_path = os.path.join(base_dir, new_name)
                    
                    try:
                        # Check if target already exists
                        if os.path.exists(new_path):
                            pbar.set_postfix_str(f"Target exists: {new_name}")
                            continue
                        
                        os.rename(old_path, new_path)
                        renamed_count += 1
                        pbar.set_postfix_str(f"✅ {folder_name} -> {new_name}")
                    except Exception as e:
                        pbar.set_postfix_str(f"❌ Error: {folder_name}")
                else:
                    pbar.set_postfix_str(f"No change needed: {folder_name}")
            else:
                pbar.set_postfix_str(f"❓ No pattern matched: {folder_name}")
    
    print(f"Chapter renaming completed! Renamed: {renamed_count} folders")
    return renamed_count

# ===== LOGIC XỬ LÝ VN1 VÀ RAW1 =====
def copy(basedir, rdir):
    """Copy và rename files từ các subfolder trong basedir vào rdir"""
    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(rdir, exist_ok=True)
    
    for fn in os.listdir(basedir):
        subdir = os.path.join(basedir, fn)
        if not os.path.isdir(subdir):
            continue
        
        # Lấy tất cả file .jpg và .png
        jpgs = glob.glob(os.path.join(subdir, '*.jpg'))
        pngs = glob.glob(os.path.join(subdir, '*.png'))
        lists = jpgs + pngs
        
        for path in lists:
            name = os.path.basename(path)
            # Làm sạch tên file
            name = name.replace(' copy', '').replace('_1', '').replace('finalSFX', '') \
                       .replace('_waiu2x_noise1_scale_x1.0', '').replace('_denoised', '').replace('f', '')
            
            # Copy với tên mới: {folder}-{filename}
            new_path = os.path.join(rdir, f'{fn}-{name}')
            try:
                shutil.copyfile(path, new_path)
            except Exception as e:
                print(f"Error copying {path}: {e}")

def rename(basedir):
    """Đổi tên folder với nhiều pattern khác nhau - phiên bản cải tiến"""
    folders = [f for f in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, f))]
    renamed_count = 0
    
    def extract_chapter_number_advanced(folder_name):
        """Extract chapter number from complex folder names (same logic as rename_chap_folders)"""
        
        # Pattern 1: cVol.XXCh.YYY.Z format
        vol_ch_match = re.search(r"c?Vol\.(?:\d+|TBD)Ch\.(\d+(?:\.\d+)?)", folder_name, re.IGNORECASE)
        if vol_ch_match:
            chap_num = vol_ch_match.group(1)
            if '.' in chap_num:
                main_part, decimal_part = chap_num.split('.')
                chap_num = str(int(main_part)) + '.' + decimal_part
            else:
                chap_num = str(int(chap_num))
            return chap_num
        
        # Pattern 2: cCh.XXX format
        ch_match = re.search(r"c?Ch\.(\d+(?:\.\d+)?)", folder_name, re.IGNORECASE)
        if ch_match:
            chap_num = ch_match.group(1)
            if '.' in chap_num:
                main_part, decimal_part = chap_num.split('.')
                chap_num = str(int(main_part)) + '.' + decimal_part
            else:
                chap_num = str(int(chap_num))
            return chap_num
        
        # Pattern 3: Standard "Chap" + số
        chap_match = re.search(r"c?Chap(?:ter)?\s*(\d+(?:\.\d+)?)", folder_name, re.IGNORECASE)
        if chap_match:
            return chap_match.group(1)
        
        # Pattern 4: Volume + Chapter format
        vol_chapter_match = re.search(r"Vol(?:ume)?\s*\d+\s*Ch(?:apter)?\s*(\d+(?:\.\d+)?)", folder_name, re.IGNORECASE)
        if vol_chapter_match:
            return vol_chapter_match.group(1)
        
        # Pattern 5: Vietnamese format
        vn_match = re.search(r"c?Ch[ươuư][ơương]ng\s*(\d+(?:\.\d+)?)", folder_name, re.IGNORECASE)
        if vn_match:
            return vn_match.group(1)
        
        # Pattern 6: Korean format
        kr_match = re.search(r"c?제\s*(\d+(?:\.\d+)?)\s*화?", folder_name, re.IGNORECASE)
        if kr_match:
            return kr_match.group(1)
        
        # Pattern 7: Already correct format
        correct_match = re.match(r"^c(\d+(?:\.\d+)?)$", folder_name, re.IGNORECASE)
        if correct_match:
            return "ALREADY_CORRECT"
        
        return None
    
    with tqdm(folders, desc="Renaming subfolders", unit="folder") as pbar:
        for fn in pbar:
            old_path = os.path.join(basedir, fn)
            pbar.set_postfix_str(f"Processing {fn}")
            
            # Try advanced pattern matching first
            chap_num = extract_chapter_number_advanced(fn)
            
            if chap_num == "ALREADY_CORRECT":
                pbar.set_postfix_str(f"Already correct: {fn}")
                continue
            elif chap_num:
                new_name = f"c{chap_num}"
            else:
                # Fallback: Old pattern matching for other formats
                new_name = fn.replace('Chapter ', 'c').replace('chapter-', 'c').replace('Chương ', 'c') \
                             .replace('Chuong', 'c').replace('chương ', 'c').replace('제 ', 'c') \
                             .replace('화', '').replace(' ', '')
                
                # Add 'c' prefix if not present
                if not new_name.startswith('c') and new_name != fn:
                    new_name = f'c{new_name}'
            
            # Only rename if different from current name
            if new_name and new_name != fn:
                new_path = os.path.join(basedir, new_name)
                
                try:
                    # Check if target already exists
                    if os.path.exists(new_path):
                        pbar.set_postfix_str(f"Target exists: {new_name}")
                        continue
                    
                    os.rename(old_path, new_path)
                    renamed_count += 1
                    pbar.set_postfix_str(f"✅ {fn} -> {new_name}")
                except Exception as e:
                    pbar.set_postfix_str(f"❌ Error: {fn}")
            else:
                pbar.set_postfix_str(f"No change needed: {fn}")
    
    if renamed_count > 0:
        print(f"Renamed {renamed_count} subfolders")

def rename1(basedir):
    """Fix duplicate 'cc' thành 'c'"""
    folders = [f for f in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, f))]
    fixed_count = 0
    
    with tqdm(folders, desc="Fixing duplicates", unit="folder") as pbar:
        for fn in pbar:
            old_path = os.path.join(basedir, fn)
            pbar.set_postfix_str(f"Checking {fn}")
            
            new_name = fn.replace('cc', 'c')
            new_path = os.path.join(basedir, new_name)
            
            if old_path != new_path:
                try:
                    os.rename(old_path, new_path)
                    fixed_count += 1
                    pbar.set_postfix_str(f"Fixed to {new_name}")
                except Exception as e:
                    pbar.set_postfix_str(f"Error: {fn}")
    
    if fixed_count > 0:
        print(f"Fixed {fixed_count} duplicate names")

# ===== MAIN PROCESSING =====
def process_vn_raw_folders(base_dir, convert_images=True):
    """Xử lý folders vn1 và raw1, tạo ra vn2 và raw2 với ảnh đã convert"""
    vn1_dir = os.path.join(base_dir, 'vn1')
    raw1_dir = os.path.join(base_dir, 'raw1')
    vn2_dir = os.path.join(base_dir, 'vn2')
    raw2_dir = os.path.join(base_dir, 'raw2')
    
    print(f"Processing VN and RAW folders in: {base_dir}")
    
    # Kiểm tra folders tồn tại
    if not os.path.exists(vn1_dir):
        print(f"Warning: {vn1_dir} does not exist")
        return
    
    if not os.path.exists(raw1_dir):
        print(f"Warning: {raw1_dir} does not exist")
        return
    
    # Tạo folders đầu ra
    os.makedirs(vn2_dir, exist_ok=True)
    os.makedirs(raw2_dir, exist_ok=True)
    print(f"Created output folders: vn2, raw2")
    
    # Đổi tên các subfolder trong vn1 và raw1 trước
    print("\n=== RENAMING SUBFOLDERS ===")
    for folder_dir in [vn1_dir, raw1_dir]:
        if os.path.exists(folder_dir):
            folder_name = os.path.basename(folder_dir)
            print(f"Processing {folder_name}:")
            rename(folder_dir)      # Đổi tên với nhiều pattern
            rename1(folder_dir)     # Fix duplicate 'cc'
    
    # Xử lý từng folder (vn1->vn2, raw1->raw2)
    folder_pairs = [(vn1_dir, vn2_dir, "VN"), (raw1_dir, raw2_dir, "RAW")]
    
    for source_dir, target_dir, folder_type in folder_pairs:
        print(f"\n=== PROCESSING {folder_type} IMAGES ===")
        
        # Copy ảnh từ subfolders trong source_dir vào target_dir với format conversion
        if convert_images:
            copy_and_convert_images(source_dir, target_dir)
        else:
            copy_images_only(source_dir, target_dir)
    
    print("\n🎉 VN and RAW folder processing completed!")

def copy_and_convert_images(source_dir, target_dir):
    """Copy và convert ảnh từ các subfolder trong source_dir vào target_dir"""
    print(f"Copying and converting images from {source_dir} to {target_dir}")
    
    valid_exts = ('.png', '.bmp', '.jpeg', '.webp', '.tiff', '.gif', '.jpg')
    converted_count = 0
    copied_count = 0
    error_count = 0
    
    # Đếm tổng số file cần xử lý
    total_files = 0
    subfolders = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(source_dir, subfolder)
        files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(valid_exts)]
        total_files += len(files)
    
    with tqdm(total=total_files, desc="Processing images", unit="img") as pbar:
        for subfolder in subfolders:
            subfolder_path = os.path.join(source_dir, subfolder)
            
            # Xử lý từng file ảnh trong subfolder
            for filename in os.listdir(subfolder_path):
                if filename.lower().endswith(valid_exts):
                    source_file = os.path.join(subfolder_path, filename)
                    pbar.set_postfix_str(f"{subfolder}/{filename}")
                    
                    try:
                        # Tạo tên file đầu ra: {subfolder}-{filename}
                        if filename.lower().endswith('.jpg'):
                            # File đã là JPG, chỉ copy
                            output_filename = f"{subfolder}-{filename}"
                            output_path = os.path.join(target_dir, output_filename)
                            shutil.copyfile(source_file, output_path)
                            copied_count += 1
                        else:
                            # Convert sang JPG
                            base_name = os.path.splitext(filename)[0]
                            output_filename = f"{subfolder}-{base_name}.jpg"
                            output_path = os.path.join(target_dir, output_filename)
                            
                            # Đọc và convert ảnh
                            img = Image.open(source_file)
                            img = img.convert("RGB")
                            img.save(output_path, "JPEG", quality=95)
                            converted_count += 1
                            
                    except Exception as e:
                        error_count += 1
                        pbar.set_postfix_str(f"Error: {filename}")
                    
                    pbar.update(1)
    
    print(f"Results: Copied {copied_count}, Converted {converted_count}, Errors {error_count}")

def copy_images_only(source_dir, target_dir):
    """Chỉ copy ảnh mà không convert"""
    print(f"Copying images from {source_dir} to {target_dir}")
    
    valid_exts = ('.png', '.bmp', '.jpeg', '.webp', '.tiff', '.gif', '.jpg')
    copied_count = 0
    error_count = 0
    
    # Đếm tổng số file cần xử lý
    total_files = 0
    subfolders = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(source_dir, subfolder)
        files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(valid_exts)]
        total_files += len(files)
    
    with tqdm(total=total_files, desc="Copying images", unit="img") as pbar:
        for subfolder in subfolders:
            subfolder_path = os.path.join(source_dir, subfolder)
            
            # Copy từng file ảnh
            for filename in os.listdir(subfolder_path):
                if filename.lower().endswith(valid_exts):
                    source_file = os.path.join(subfolder_path, filename)
                    output_filename = f"{subfolder}-{filename}"
                    output_path = os.path.join(target_dir, output_filename)
                    pbar.set_postfix_str(f"{subfolder}/{filename}")
                    
                    try:
                        shutil.copyfile(source_file, output_path)
                        copied_count += 1
                    except Exception as e:
                        error_count += 1
                        pbar.set_postfix_str(f"Error: {filename}")
                    
                    pbar.update(1)
    
    print(f"Results: Copied {copied_count}, Errors {error_count}")

if __name__ == "__main__":
    # Test cases for the new renaming function
    def test_rename_patterns():
        """Test the new renaming patterns with provided examples"""
        test_cases = [
            "cCh.293",           # Expected: c293
            "cVol.02Ch.001.5",   # Expected: c1.5
            "cVol.3Ch.20",       # Expected: c20
            "cVol.21Ch.192.1",   # Expected: c192.1
            "cVol.TBDCh.044.5",  # Expected: c44.5
            "Chapter 25",        # Expected: c25
            "Chương 10",         # Expected: c10
            "제 5화",             # Expected: c5
            "c123",              # Expected: Already correct
            "c45.5",             # Expected: Already correct
        ]
        
        print("🧪 Testing rename patterns:")
        print("=" * 50)
        
        from gendata import rename_chap_folders
        import tempfile
        import os
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test folders
            for test_case in test_cases:
                test_folder = os.path.join(temp_dir, test_case)
                os.makedirs(test_folder, exist_ok=True)
            
            print(f"📁 Before renaming:")
            for folder in sorted(os.listdir(temp_dir)):
                print(f"   {folder}")
            
            print("\n🔄 Running rename function...")
            
            # Extract chapter number function (copy from the main function)
            def extract_chapter_number(folder_name):
                import re
                
                # Pattern 1: cVol.XXCh.YYY.Z format
                vol_ch_match = re.search(r"c?Vol\.(?:\d+|TBD)Ch\.(\d+(?:\.\d+)?)", folder_name, re.IGNORECASE)
                if vol_ch_match:
                    chap_num = vol_ch_match.group(1)
                    if '.' in chap_num:
                        main_part, decimal_part = chap_num.split('.')
                        chap_num = str(int(main_part)) + '.' + decimal_part
                    else:
                        chap_num = str(int(chap_num))
                    return chap_num
                
                # Pattern 2: cCh.XXX format
                ch_match = re.search(r"c?Ch\.(\d+(?:\.\d+)?)", folder_name, re.IGNORECASE)
                if ch_match:
                    chap_num = ch_match.group(1)
                    if '.' in chap_num:
                        main_part, decimal_part = chap_num.split('.')
                        chap_num = str(int(main_part)) + '.' + decimal_part
                    else:
                        chap_num = str(int(chap_num))
                    return chap_num
                
                # Other patterns...
                chap_match = re.search(r"c?Chap(?:ter)?\s*(\d+(?:\.\d+)?)", folder_name, re.IGNORECASE)
                if chap_match:
                    return chap_match.group(1)
                
                vn_match = re.search(r"c?Ch[ươuư][ơương]ng\s*(\d+(?:\.\d+)?)", folder_name, re.IGNORECASE)
                if vn_match:
                    return vn_match.group(1)
                
                kr_match = re.search(r"c?제\s*(\d+(?:\.\d+)?)\s*화?", folder_name, re.IGNORECASE)
                if kr_match:
                    return kr_match.group(1)
                
                correct_match = re.match(r"^c(\d+(?:\.\d+)?)$", folder_name, re.IGNORECASE)
                if correct_match:
                    return "ALREADY_CORRECT"
                
                return None
            
            # Test each case
            print("\n📋 Test results:")
            for original in test_cases:
                chap_num = extract_chapter_number(original)
                if chap_num == "ALREADY_CORRECT":
                    result = "Already correct"
                elif chap_num:
                    result = f"c{chap_num}"
                else:
                    result = "No match"
                
                print(f"   {original:20} → {result}")
        
        print("\n" + "=" * 50)
    
    # Run test
    print("🧪 Running rename pattern tests...")
    test_rename_patterns()
    
    # Original main function
    print("\n🚀 Starting actual processing...")
    
    # Thay đường dẫn này thành folder chứa vn1 và raw1
    base_dir = r"f:\manhwa\Nhiem_vu_doi_that"
    
    print("🚀 Starting manga processing pipeline...")
    print(f"Base directory: {base_dir}")
    print("=" * 60)
    
    print("\n=== CHAPTER FOLDER RENAMING ===")
    rename_chap_folders(base_dir)
    
    print("\n=== VN1 AND RAW1 PROCESSING ===")
    # convert_images=True để chuyển đổi ảnh sang JPG, False để giữ nguyên format
    process_vn_raw_folders(base_dir, convert_images=True)
    
    print("\n" + "=" * 60)
    print("🎉 All processing completed successfully!")
