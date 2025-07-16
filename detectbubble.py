import os
import sys
import requests
import torch
import numpy as np
from ultralytics import YOLO
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime
from time import time

# Đường dẫn đến các thư mục chứa ảnh
result_dir = r"f:\Manhwa_result\zfinish\hunter-academys-battle-god-961\inpainted"  # Ảnh tiếng Anh (để phát hiện bong bóng)
vn_dir = r"f:\Manhwa_result\zfinish\hunter-academys-battle-god-961"  # Ảnh tiếng Việt (nguồn lấy nội dung)

# Thư mục đầu ra - chỉ lưu kết quả cuối cùng
output_dir = r"f:\Manhwa_result\zfinish\hunter-academys-battle-god-961\result"
os.makedirs(output_dir, exist_ok=True)

# Đường dẫn đến mô hình
model_dir = "yolo"
model_path = os.path.join(model_dir, "best.pt")

# Tạo thư mục log để ghi lại quá trình xử lý
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"processing_log_{time():.0f}.txt")

def log_message(message):
    """Ghi log vào file và hiển thị trên console"""
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{message}\n")

# Hàm tải xuống mô hình
def download_file(url, destination):
    log_message(f"Đang tải xuống mô hình từ {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
        
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))
        
        progress_bar.close()
        
        # Kiểm tra kích thước tệp đã tải
        file_size = os.path.getsize(destination)
        if file_size < 1000000:  # Nếu tệp nhỏ hơn 1MB, có thể đã tải không đầy đủ
            log_message(f"Cảnh báo: Tệp đã tải có kích thước nhỏ ({file_size} bytes). Có thể đã tải không đầy đủ.")
            return False
        return True
    except Exception as e:
        log_message(f"Lỗi khi tải xuống: {e}")
        return False

# def check_image_size(image_path):
#     """Kiểm tra kích thước ảnh"""
#     try:
#         img = cv2.imread(image_path)
#         if img is None:
#             return None
#         height, width = img.shape[:2]
#         return width, height
#     except Exception as e:
#         log_message(f"Lỗi khi kiểm tra kích thước ảnh {image_path}: {e}")
#         return None

def detect_bubbles_with_sliding_window(model, image_path, window_size=1024, overlap=256, conf=0.25):
    """
    Phát hiện bong bóng văn bản trong ảnh lớn sử dụng kỹ thuật sliding window
    
    Args:
        model: Mô hình YOLO đã tải
        image_path: Đường dẫn đến ảnh cần phát hiện
        window_size: Kích thước cửa sổ trượt
        overlap: Độ chồng lấp giữa các cửa sổ
        conf: Ngưỡng tin cậy cho các dự đoán
        
    Returns:
        list: Danh sách các bong bóng được phát hiện sau khi hợp nhất [x1, y1, x2, y2, confidence]
    """
    # Đọc ảnh gốc
    img = cv2.imread(image_path)
    if img is None:
        log_message(f"Không thể đọc ảnh: {image_path}")
        return []
    
    # Lấy kích thước ảnh
    height, width = img.shape[:2]
    log_message(f"Phát hiện bong bóng trong ảnh {os.path.basename(image_path)} với kích thước {width}x{height}")
    
    # Tạo một danh sách để lưu tất cả các dự đoán
    all_boxes = []
    all_confs = []
    
    # Tính số lượng cửa sổ trượt
    num_windows_y = max(1, (height - overlap) // (window_size - overlap))
    num_windows_x = max(1, (width - overlap) // (window_size - overlap))
    total_windows = num_windows_y * num_windows_x
    
    log_message(f"Chia ảnh thành {total_windows} cửa sổ ({num_windows_x}x{num_windows_y}) với kích thước {window_size} và độ chồng lấp {overlap}")
    
    # Theo dõi tiến trình
    window_progress = tqdm(total=total_windows, desc="Phát hiện bong bóng")
    
    # Xử lý ảnh theo các cửa sổ trượt
    for y_start in range(0, height, window_size - overlap):
        # Đảm bảo cửa sổ không vượt quá kích thước ảnh
        y_end = min(y_start + window_size, height)
        if y_end == height:
            y_start = max(0, y_end - window_size)
        
        for x_start in range(0, width, window_size - overlap):
            x_end = min(x_start + window_size, width)
            if x_end == width:
                x_start = max(0, x_end - window_size)
            
            # Tạo một cửa sổ trượt
            window = img[y_start:y_end, x_start:x_end].copy()
            
            try:
                # Dự đoán trên cửa sổ trượt
                results = model(window, conf=conf, verbose=False)
                result = results[0]
                
                # Kiểm tra xem có phát hiện không
                if len(result.boxes) > 0:
                    # Lưu lại các dự đoán để xử lý sau
                    boxes = result.boxes.xyxy.cpu().numpy()  
                    confs = result.boxes.conf.cpu().numpy()
                    
                    # Điều chỉnh tọa độ các box về hệ tọa độ của ảnh gốc
                    for i, box in enumerate(boxes):
                        adjusted_box = box.copy()
                        # Điều chỉnh tọa độ
                        adjusted_box[0] += x_start
                        adjusted_box[1] += y_start
                        adjusted_box[2] += x_start
                        adjusted_box[3] += y_start
                        all_boxes.append(adjusted_box)
                        all_confs.append(confs[i])
            except Exception as e:
                log_message(f"Lỗi khi xử lý cửa sổ ({x_start},{y_start})-({x_end},{y_end}): {e}")
            
            window_progress.update(1)
    
    window_progress.close()
    
    # Hợp nhất các dự đoán chồng lấp sử dụng NMS (Non-Maximum Suppression)
    if all_boxes:
        log_message(f"Phát hiện tổng cộng {len(all_boxes)} bong bóng văn bản trước khi hợp nhất")
        
        # Chuyển danh sách thành numpy arrays
        all_boxes = np.array(all_boxes)
        all_confs = np.array(all_confs)
        
        # Sử dụng NMS để loại bỏ các dự đoán chồng lấp
        try:
            from ultralytics.utils.ops import non_max_suppression
            # Chuyển về định dạng torch
            boxes_tensor = torch.from_numpy(all_boxes).float()
            confs_tensor = torch.from_numpy(all_confs).float()
            
            # Chuẩn bị đầu vào cho NMS
            boxes_with_confs = torch.cat([boxes_tensor, confs_tensor.unsqueeze(-1)], dim=1)
            
            # Áp dụng NMS
            nms_threshold = 0.5  # Ngưỡng cho IOU
            keep_indices = non_max_suppression(
                boxes_with_confs,
                conf_thres=conf,
                iou_thres=nms_threshold
            )
            
            # Lấy các boxes sau khi áp dụng NMS
            merged_boxes = []
            for boxes in keep_indices:
                for box in boxes:
                    x1, y1, x2, y2, conf_val = box.cpu().numpy()
                    merged_boxes.append([int(x1), int(y1), int(x2), int(y2), float(conf_val)])
            
            log_message(f"Sau khi hợp nhất, còn lại {len(merged_boxes)} bong bóng văn bản")
            return merged_boxes
        
        except Exception as e:
            log_message(f"Lỗi khi áp dụng NMS: {e}")
            # Nếu có lỗi, trả về các box chưa qua xử lý NMS
            merged_boxes = []
            for i in range(len(all_boxes)):
                box = all_boxes[i]
                conf = all_confs[i]
                merged_boxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(conf)])
            return merged_boxes
    else:
        # Không có phát hiện
        log_message(f"Không phát hiện bong bóng văn bản trong {os.path.basename(image_path)}")
        return []

def transfer_bubble_content(result_img, vn_img, bubbles):
    """
    Chuyển nội dung bong bóng văn bản từ ảnh tiếng Việt sang ảnh tiếng Anh
    
    Args:
        result_img: Ảnh tiếng Anh (numpy array)
        vn_img: Ảnh tiếng Việt (numpy array)
        bubbles: Danh sách các bong bóng [x1, y1, x2, y2, confidence]
        
    Returns:
        np.ndarray: Ảnh kết quả sau khi chuyển nội dung
    """
    # Tạo ảnh kết quả từ ảnh tiếng Anh
    output_img = result_img.copy()
    
    # Xử lý từng bong bóng
    for bubble in bubbles:
        x1, y1, x2, y2 = map(int, bubble[:4])
        
        # Kiểm tra kích thước bong bóng
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue
        
        # Đảm bảo tọa độ nằm trong ảnh
        height, width = result_img.shape[:2]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        
        # Lấy vùng tương ứng trên ảnh tiếng Việt
        vn_bubble_region = vn_img[y1:y2, x1:x2]
        
        # Gán nội dung tiếng Việt vào ảnh kết quả
        output_img[y1:y2, x1:x2] = vn_bubble_region
    
    return output_img

def match_image_heights(result_img, vn_img):
    """
    So sánh kích thước của hai ảnh và thực hiện padding nếu cần
    để đảm bảo chiều cao (height) của hai ảnh bằng nhau.
    
    Args:
        result_img: Ảnh tiếng Anh (numpy array)
        vn_img: Ảnh tiếng Việt (numpy array)
        
    Returns:
        tuple: (result_img, vn_img) đã được điều chỉnh
    """
    # Lấy kích thước của hai ảnh
    result_height, result_width = result_img.shape[:2]
    vn_height, vn_width = vn_img.shape[:2]
    
    # Nếu chiều cao đã bằng nhau, không cần xử lý thêm
    if result_height == vn_height:
        return result_img, vn_img
    
    # Xác định ảnh nào cần padding
    if result_height < vn_height:
        # Padding cho ảnh tiếng Anh
        pad_height = vn_height - result_height
        # Tạo vùng padding với màu trắng (255,255,255)
        padding = np.ones((pad_height, result_width, 3), dtype=np.uint8) * 255
        # Ghép padding vào cuối ảnh
        padded_result_img = np.vstack([result_img, padding])
        return padded_result_img, vn_img
    else:
        # Padding cho ảnh tiếng Việt
        pad_height = result_height - vn_height
        # Tạo vùng padding với màu trắng (255,255,255)
        padding = np.ones((pad_height, vn_width, 3), dtype=np.uint8) * 255
        # Ghép padding vào cuối ảnh
        padded_vn_img = np.vstack([vn_img, padding])
        return result_img, padded_vn_img

def process_bubble_transfer():
    """
    Xử lý chuyển nội dung bong bóng từ ảnh tiếng Việt sang ảnh tiếng Anh
    sử dụng phương pháp sliding window để phát hiện bong bóng
    """
    global model_path

    start_time = datetime.now()
    log_message(f"Bắt đầu xử lý lúc {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Kiểm tra mô hình đã tồn tại chưa
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
        log_message("Mô hình chưa có hoặc bị hỏng, đang tải xuống...")
        model_url = "https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble/resolve/main/model.pt"
        success = download_file(model_url, model_path)
        if not success:
            log_message("Không thể tải xuống mô hình từ URL.")
            try:
                from huggingface_hub import hf_hub_download
                log_message("Đang thử tải xuống bằng Hugging Face Hub API...")
                downloaded_path = hf_hub_download(
                    repo_id="kitsumed/yolov8m_seg-speech-bubble",
                    filename="model.pt",
                    cache_dir=model_dir
                )
                # Cập nhật đường dẫn model_path nếu tải thành công qua API
                model_path = downloaded_path
                log_message(f"Đã tải xuống mô hình thành công vào: {model_path}")
            except Exception as e:
                log_message(f"Lỗi khi tải mô hình qua Hugging Face API: {e}")
                log_message("Vui lòng tải xuống mô hình thủ công từ: https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble")
                log_message(f"Sau đó đặt tệp 'model.pt' vào thư mục '{model_dir}'")
                return
    
    # Tải mô hình với verbose=False để giảm log
    try:
        model = YOLO(model_path, verbose=False)
        log_message("Đã tải mô hình thành công")
    except Exception as e:
        log_message(f"Lỗi khi tải mô hình: {e}")
        return
    
    # Lấy danh sách các file ảnh
    result_files = [f for f in os.listdir(result_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not result_files:
        log_message(f"Không tìm thấy file ảnh nào trong thư mục {result_dir}")
        return
    
    log_message(f"Tìm thấy {len(result_files)} ảnh để xử lý")
    
    processed_count = 0
    detected_count = 0
    
    # Thiết lập tham số cho sliding window
    window_size = 1024  # Kích thước cửa sổ
    overlap = 256       # Độ chồng lấp giữa các cửa sổ
    conf = 0.25         # Ngưỡng tin cậy
    
    # Xử lý từng ảnh
    for filename in tqdm(result_files, desc="Đang xử lý ảnh"):
        # Đường dẫn đến các file
        result_path = os.path.join(result_dir, filename)  # Ảnh tiếng Anh
        vn_path = os.path.join(vn_dir, filename)  # Ảnh tiếng Việt
        
        # Đường dẫn đến file đầu ra (lưu trực tiếp vào thư mục result)
        output_path = os.path.join(output_dir, filename)  # Ảnh kết quả
        
        # Kiểm tra file ảnh tiếng Việt có tồn tại không
        if not os.path.exists(vn_path):
            log_message(f"Không tìm thấy ảnh tiếng Việt tương ứng cho {filename}")
            continue
        
        try:
            # Đọc ảnh tiếng Anh và tiếng Việt
            result_img = cv2.imread(result_path)
            vn_img = cv2.imread(vn_path)
            
            if result_img is None or vn_img is None:
                log_message(f"Không thể đọc ảnh tiếng Anh hoặc tiếng Việt của {filename}")
                continue
            
            # Kiểm tra kích thước của hai ảnh
            if result_img.shape != vn_img.shape:
                log_message(f"Ảnh tiếng Anh và tiếng Việt có kích thước khác nhau: {filename}")
                log_message(f"Tiếng Anh: {result_img.shape}, Tiếng Việt: {vn_img.shape}")
                
                # So sánh chiều rộng
                if result_img.shape[1] != vn_img.shape[1]:
                    # Nếu chiều rộng khác nhau, resize để có cùng chiều rộng
                    log_message("Điều chỉnh chiều rộng để khớp nhau")
                    target_width = min(result_img.shape[1], vn_img.shape[1])
                    if result_img.shape[1] != target_width:
                        result_img = cv2.resize(result_img, (target_width, result_img.shape[0]))
                    if vn_img.shape[1] != target_width:
                        vn_img = cv2.resize(vn_img, (target_width, vn_img.shape[0]))
                
                # Sau đó kiểm tra và thực hiện padding cho chiều cao nếu cần
                result_img, vn_img = match_image_heights(result_img, vn_img)
                log_message(f"Đã điều chỉnh kích thước và padding: Tiếng Anh: {result_img.shape}, Tiếng Việt: {vn_img.shape}")
            
            # Phát hiện bong bóng văn bản trên ảnh tiếng Anh sử dụng sliding window
            bubbles = detect_bubbles_with_sliding_window(
                model=model,
                image_path=result_path,
                window_size=window_size,
                overlap=overlap,
                conf=conf
            )
            
            if not bubbles:
                log_message(f"Không phát hiện bong bóng văn bản trong {filename}")
                # Nếu không phát hiện bong bóng, sao chép ảnh gốc
                cv2.imwrite(output_path, result_img)
                continue
            
            detected_count += 1
            log_message(f"Phát hiện {len(bubbles)} bong bóng văn bản trong {filename}")
            
            # Chuyển nội dung bong bóng từ ảnh tiếng Việt sang ảnh tiếng Anh
            output_img = transfer_bubble_content(result_img, vn_img, bubbles)
            
            # Lưu ảnh kết quả vào thư mục output
            cv2.imwrite(output_path, output_img)
            
            processed_count += 1
            log_message(f"Đã xử lý thành công ảnh {filename}")
            
        except Exception as e:
            log_message(f"Lỗi khi xử lý ảnh {filename}: {e}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    log_message(f"\n✅ Hoàn tất xử lý tất cả ảnh trong {int(hours)} giờ {int(minutes)} phút {seconds:.2f} giây")
    log_message(f"Tổng số ảnh: {len(result_files)}")
    log_message(f"Số ảnh được xử lý: {processed_count}")
    log_message(f"Số ảnh có bong bóng được phát hiện: {detected_count}")
    log_message(f"Kết quả được lưu tại: {output_dir}")

if __name__ == "__main__":
    process_bubble_transfer()