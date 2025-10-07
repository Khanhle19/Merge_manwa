import os
from PIL import Image

def convert_images_in_all_subfolders(root_dir):
    # Các định dạng ảnh phổ biến
    valid_exts = ('.png', '.bmp', '.jpeg', '.webp', '.tiff', '.gif', '.jpg')
    # Duyệt qua tất cả các thư mục con trong root_dir
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

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
                    # Nếu file đã là .jpg thì chỉ kiểm tra, có thể bỏ qua hoặc ghi đè nếu muốn
                    if filename.lower().endswith('.jpg'):
                        continue 
                    # Lưu ảnh dưới dạng JPG với chất lượng cao
                    img = img.convert("RGB")
                    img.save(new_file_path, "JPEG", quality=95)
                    print(f"✅ Converted: {file_path} -> {new_file_path}")
                    # Xóa file cũ nếu muốn
                    os.remove(file_path)
                except Exception as e:
                    print(f"❌ Error converting {file_path}: {e}")

if __name__ == "__main__":
    root_dir = r"H:\manhwa\The_Martial_God_Who_Regressed_Back_to_Level_2\vn1"
    convert_images_in_all_subfolders(root_dir)